/****************************************************************************/
/****************************************************************************/
/**                                                                        **/
/**                TU Muenchen - Institut fuer Informatik                  **/
/**                                                                        **/
/** Copyright: Prof. Dr. Thomas Ludwig                                     **/
/**            Andreas C. Schmidt                                          **/
/**            JK und andere  besseres Timing, FLOP Berechnung             **/
/**                                                                        **/
/** File:      partdiff-seq.c                                              **/
/**                                                                        **/
/** Purpose:   Partial differential equation solver for Gauss-Seidel and   **/
/**            Jacobi method.                                              **/
/**                                                                        **/
/****************************************************************************/
/****************************************************************************/

/* ************************************************************************ */
/* Include standard header file.                                            */
/* ************************************************************************ */
#define _POSIX_C_SOURCE 200809L

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>
#include <malloc.h>
#include <sys/time.h>

#include "partdiff-seq.h"

//Änderungen am Programm sind jeweils von 2 Kommentaren //Geändert umschlossen

struct calculation_arguments
{
	uint64_t  N;              /* number of spaces between lines (lines=N+1)     */
	uint64_t  num_matrices;   /* number of matrices                             */
	double    h;              /* length of a space between two lines            */
	double    ***Matrix;      /* index matrix used for addressing M             */
	double    *M;             /* two matrices with real values                  */
	//Geändert
	uint64_t  firstRow;		  /* global Index of the local Index 0				*/
	uint64_t  rows;			  /* Number of local rows							*/
	int		  rank;		   	  /* Rank of local process							*/
	int		  num_processes;  /* Number of used processes						*/
	//Geändert
};

struct calculation_results
{
	uint64_t  m;
	uint64_t  stat_iteration; /* number of current iteration                    */
	double    stat_precision; /* actual precision of all slaves in iteration    */
};

/* ************************************************************************ */
/* Global variables                                                         */
/* ************************************************************************ */

/* time measurement variables */
struct timeval start_time;       /* time when program started                      */
struct timeval comp_time;        /* time when calculation completed                */


/* ************************************************************************ */
/* initVariables: Initializes some global variables                         */
/* ************************************************************************ */
static
void
initVariables (struct calculation_arguments* arguments, struct calculation_results* results, struct options const* options)
{
	arguments->N = (options->interlines * 8) + 9 - 1;
	arguments->num_matrices = (options->method == METH_JACOBI) ? 2 : 1;
	arguments->h = 1.0 / arguments->N;
	//Geändert
	/*Berechne die Anzahl an Zeilen pro Prozess. rows gibt nur die Anzahl an Zeilen auf denen der Prozess wirklich rechnet wieder. */
	int rank, num_tasks;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);
	div_t d = div(arguments->N+1, num_tasks);   //Zeilen/(Anzahl Prozessoren)  Gespeichert als Quotient und Rest
												//N ist Indexbehaftet also für Matrixzugriff von 0 bis (echte Zeilen-1)
	unsigned numLines = d.quot; 			    //Jeder Prozess erhält zunächst GesamtZeilen/ProzessAnzahl abgerundet
	unsigned firstRow = rank * d.quot; 			//FirstRow stellt die erste Zeile auf der der Prozess wirklich rechnet in Bezug auf die gesamt Matrix dar
	if(rank < d.rem) //Wenn es einen Rest gibt also nicht jeder Prozess die gleiche Anzahl Zeilen berechnen kann
	{
		numLines++;    //Erhalten die ersten n Prozesse eine extra Zeile. Wobei n dem Rest entspricht
		firstRow = firstRow + rank;    //FirstRow verschiebt sich jeweils um Anzahl der Vorgänger Prozesse mit extra Zeile. Das entspricht genau dem eigenen Rank
	}
	else
	{
		firstRow = firstRow + d.rem;   //Wenn Rang höher als Rest gab es bereits N=Rest Prozesse mit extra Zeile
	}
	if(rank == 0) //Spezialbehandlung des 1. und letzten Ranks aufgrund der Ränder der Matrix
	{
		firstRow = 1;    //rechnet erst wirklich auf Zeile eins Aufgrund des oberen Randes
		numLines--;		 //Damit hat der Prozess auch 1 Zeile weniger auf der er wirklich rechnet
	}
	if(rank == num_tasks-1)   //Letzter Rank: Ähnlich nur das firstRow stimmt aber auch hier eine Zeile weniger berechnet wird(wegen unterem Rand der Matrix)
	{
		numLines--;
	}
	arguments->firstRow = firstRow;
	arguments->rows = numLines;
	arguments->rank = rank;
	arguments->num_processes = num_tasks;
	//Geändert

	results->m = 0;
	results->stat_iteration = 0;
	results->stat_precision = 0;
}

/* ************************************************************************ */
/* freeMatrices: frees memory for matrices                                  */
/* ************************************************************************ */
static
void
freeMatrices (struct calculation_arguments* arguments)
{
	uint64_t i;

	for (i = 0; i < arguments->num_matrices; i++)
	{
		free(arguments->Matrix[i]);
	}

	free(arguments->Matrix);
	free(arguments->M);
}

/* ************************************************************************ */
/* allocateMemory ()                                                        */
/* allocates memory and quits if there was a memory allocation problem      */
/* ************************************************************************ */
static
void*
allocateMemory (size_t size)
{
	void *p;

	if ((p = malloc(size)) == NULL)
	{
		printf("Speicherprobleme! (%" PRIu64 " Bytes)\n", size);
		/* exit program */
		exit(1);
	}

	return p;
}

/* ************************************************************************ */
/* allocateMatrices: allocates memory for matrices                          */
/* ************************************************************************ */
static
void
allocateMatrices (struct calculation_arguments* arguments)
{
	uint64_t i, j;

	uint64_t const N = arguments->N;
	uint64_t const rows = arguments->rows;

	//Geändert
	arguments->M = allocateMemory(arguments->num_matrices * (N + 2) * (rows + 2) * sizeof(double));  //Braucht nur noch jeweils rows allokieren
																									//+2 um die benachbarten Zeilen aufzunehmen
	//Geändert
	arguments->Matrix = allocateMemory(arguments->num_matrices * sizeof(double**));

	for (i = 0; i < arguments->num_matrices; i++)
	{
		arguments->Matrix[i] = allocateMemory((N + 1) * sizeof(double*));

		for (j = 0; j <= N; j++)
		{
			arguments->Matrix[i][j] = arguments->M + (i * (N + 1) * (N + 1)) + (j * (N + 1));
		}
	}
}

/* ************************************************************************ */
/* initMatrices: Initialize matrix/matrices and some global variables       */
/* ************************************************************************ */
static
void
initMatrices (struct calculation_arguments* arguments, struct options const* options)
{
	//Geändert
	uint64_t g, i, j,globali;                                /*  local variables for loops   */
	//Geändert

	uint64_t const N = arguments->N;
	double const h = arguments->h;
	double*** Matrix = arguments->Matrix;
	//Geändert
	uint64_t const rows = arguments->rows;
	uint64_t const firstRow = arguments->firstRow;
	/* initialize matrix/matrices with zeros */
	for (g = 0; g < arguments->num_matrices; g++)
	{
		for (i = 0; i < rows+2; i++)   //Matrizen der Prozesse sind jeweils um 2 größer als die Zeilen auf denen sie wirklich rechnen
									   //Bei der Initialisierung können/sollten trotzdem alle Zeilen der Matrizen initialisiert werden
		{
			for (j = 0; j <= N; j++)
			{
				Matrix[g][i][j] = 0.0;
			}
		}
	}

	/* initialize borders, depending on function (function 2: nothing to do) */
	if (options->inf_func == FUNC_F0)
	{
		for (g = 0; g < arguments->num_matrices; g++)
		{

				for (i = 0; i < rows+2; i++)  //Wie oben 2 Zeilen größer als rows und alle Zeilen selber initialisieren
				{
					globali=firstRow+i-1;   //Globali stellt den Index auf der Gesamtmatrix dar. firstRow stellt erste Zeile auf der wirklich gerechnet wird dar
											//Bei der Initialisierung wird aber von der Zeile davor angefangen (Nachbarprozessreihen in der eigenen Matrix)
											//Deswegen muss firstRow-1 gerechnet werden
					Matrix[g][i][0] = 1.0 - (h * globali);
					Matrix[g][i][N] = h * globali;
				}
			if(arguments->rank==(arguments->num_processes-1)) //Sonderbehandlung wegen unterstem Rand. Betrifft nur letzten Rank
			{
				for(i = 0; i <= N; i++)
				{
					Matrix[g][rows+1][i] = h * i;
				}
				Matrix[g][rows+1][0] = 0.0;
			}
			if(arguments->rank==0)  //Sonderbehandlung wegen oberstem Rand. Betrifft nur 1 Rank
			{
				for(i = 0; i <= N; i++)
				{
					Matrix[g][0][i] = 1.0 - (h * i);
				}
				Matrix[g][0][N] = 0.0;
			}
		}
	}
	//Geändert
}

/* ************************************************************************ */
/* calculate: solves the equation                                           */
/* ************************************************************************ */
static
void
calculate (struct calculation_arguments const* arguments, struct calculation_results *results, struct options const* options)
{
	//Geändert
	int i, j, globali;                          /* local variables for loops  */
	//Geändert
	int m1, m2;                                 /* used as indices for old and new matrices       */
	double star;                                /* four times center value minus 4 neigh.b values */
	double residuum;                            /* residuum of current iteration                  */
	double maxresiduum;                         /* maximum residuum value of a slave in iteration */

	int const N = arguments->N;
	double const h = arguments->h;
	//Geändert
	int const rows = arguments->rows;
	int const rank = arguments->rank;
	int const firstrow = arguments->firstRow;
	int const lastRank = arguments->num_processes-1;
	int const firstRank = 0;
	MPI_Request sendRequest, recvRequest, precrecvRequest;
	MPI_Status status;
	int stopflag = 0;
	//Geändert

	double pih = 0.0;
	double fpisin = 0.0;

	int term_iteration = options->term_iteration;

	/* initialize m1 and m2 depending on algorithm */
	if (options->method == METH_JACOBI)
	{
		m1 = 0;
		m2 = 1;
	}
	else
	{
		m1 = 0;
		m2 = 0;
	}

	if (options->inf_func == FUNC_FPISIN)
	{
		pih = PI * h;
		fpisin = 0.25 * TWO_PI_SQUARE * h * h;
	}
	//Geändert
	if(rank<lastRank && options->termination == TERM_PREC)   //Bei Abbruch nach Genauigkeit
	{
		MPI_Irecv(&stopflag,1,MPI_INT,lastRank,42,MPI_COMM_WORLD, &precrecvRequest);  //Alle Ranks ausser dem letzten können jederzeit Stopnachricht erhalten
	}

	while (term_iteration > 0)
	{
		double** Matrix_Out = arguments->Matrix[m1];
		double** Matrix_In  = arguments->Matrix[m2];

		if(rank==0) //Nur der erste Rank setzt jede Iteration maxresiduum = 0.
					//Alle anderen erhalten es vom Vorgängerprozess um das maxresiduum über die gesamte Tabelle zu bilden
		{
			maxresiduum = 0;
		}
		if(stopflag == 1)   //Wenn die Stopnachricht kam setzt term_iteration auf 0 und beende somit die while Schleife
		{
				term_iteration = 0;
				stopflag = 2;
		}

		if(rank>firstRank)
		{
			MPI_Recv(Matrix_Out[0],N+3,MPI_DOUBLE, rank-1,0,MPI_COMM_WORLD, &status); //Alle Prozesse außer dem 1. erwarten vor ihrer Berechnung die letzte Zeile des Vorgängerprozess
			if(options->termination == TERM_PREC || term_iteration ==1)
			{
				MPI_Recv(&maxresiduum,1,MPI_DOUBLE,rank-1,1,MPI_COMM_WORLD, &status);	//Bei Abbruch nach Genauigkeit oder in der letzten Iteration kommuniziere zusätzlich über das maxresiduum
			}
		}
		if(!(results->stat_iteration == 0) & (rank<lastRank)) //Erhalten der 1. Zeile des unteren Prozesses kann jederzeit während berechnung geschehen.
		{
			MPI_Irecv(Matrix_Out[rows+1],N+3,MPI_DOUBLE, rank+1,2,MPI_COMM_WORLD, &recvRequest);  //Achtung erst in der 2. Iteration auszuführen (Iteration bezieht sich auf Iterationen der while-Schleife)
		}

		/* over all rows */
		for (i = 1; i <= rows; i++)
		{
			double fpisin_i = 0.0;
			if((rank>firstRank) & (i == 2)) //Alle bis auf dem 1. Rank während der Berechnung ihrer 2. Zeile die 1. bereits berechnete Zeile versenden
			{
				MPI_Isend(Matrix_Out[1],N+3,MPI_DOUBLE, rank-1, 2, MPI_COMM_WORLD, &sendRequest);
			}

			if (options->inf_func == FUNC_FPISIN)
			{
				globali = firstrow+i-1; //globali wieder Index der gesamten Matrix
				fpisin_i = fpisin * sin(pih * (double)globali);
			}
			if((i==rows) & (rank<lastRank) & !(results->stat_iteration == 0) & (stopflag == 0))
			{
				MPI_Wait(&recvRequest, MPI_STATUS_IGNORE); //Vor der Berechnung der letzten Zeile auf 1. des Nachfolgerprozess warten
															//Aber erst ab der 1. Iteration
			}

			//Nicht geändert
			/* over all columns */
			for (j = 1; j < N; j++)
			{
				star = 0.25 * (Matrix_In[i-1][j] + Matrix_In[i][j-1] + Matrix_In[i][j+1] + Matrix_In[i+1][j]);

				if (options->inf_func == FUNC_FPISIN)
				{
					star += fpisin_i * sin(pih * (double)j);
				}

				if (options->termination == TERM_PREC || term_iteration == 1)
				{
					residuum = Matrix_In[i][j] - star;
					residuum = (residuum < 0) ? -residuum : residuum;
					maxresiduum = (residuum < maxresiduum) ? maxresiduum : residuum;
				}

				Matrix_Out[i][j] = star;
			}
			//Nicht geändert

			//Ab hier wieder geändert
		}
		//Nach jedem Durchlauf die letzte Zeile nach unten senden (außer letzter Rank)
		if(rank<lastRank)
		{
			MPI_Send(Matrix_Out[rows], N+3, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD);
			if(options->termination == TERM_PREC || term_iteration == 1)  //Bei Abbruch nach Genauigkeit oder letzter Iteration auch maxresiduum
			{
				MPI_Send(&maxresiduum,1,MPI_DOUBLE,rank+1,1,MPI_COMM_WORLD);
			}
		}

		results->stat_iteration++;

		/* exchange m1 and m2 */
		i = m1;
		m1 = m2;
		m2 = i;

		/* check for stopping calculation, depending on termination method */
		if (options->termination == TERM_PREC)
		{
			if ((rank == lastRank) & (maxresiduum < options->term_precision) & (stopflag == 0)) //Letzter Rank ich für das überprüfen des maxresiduum verantwortlich
			{
				stopflag = 1;
				for(int r = 0; r < lastRank; r++)
				{
					MPI_Send(&stopflag, 1, MPI_INT, r, 42, MPI_COMM_WORLD);		//Sende allen die Stopflag
				}
			}
		}
		else if (options->termination == TERM_ITER)
		{
			term_iteration--;
		}
	}
	//Nachdem alle Iterationen abgeschlossen sind sendet der letzte Prozesse dem 1. Prozess das maxresiduum damit dieser dies später ausgeben kann
	if(rank == lastRank)
	{
		MPI_Send(&maxresiduum,1,MPI_DOUBLE,firstRank,0,MPI_COMM_WORLD);
	}
	if(rank == firstRank)
	{
		MPI_Recv(&maxresiduum,1,MPI_DOUBLE,lastRank,0,MPI_COMM_WORLD, &status);
		results->stat_precision = maxresiduum;   //1. Prozess speichert in result maxresiduum für die Ausgabe
	}
	//Geändert
	results->m = m2;
}

/* ************************************************************************ */
/*  displayStatistics: displays some statistics about the calculation       */
/* ************************************************************************ */
static
void
displayStatistics (struct calculation_arguments const* arguments, struct calculation_results const* results, struct options const* options)
{
	if(arguments->rank==0)  //Faulheitsvariante bei der hier überprüft wird wer für die Ausgabe zuständig ist
	{
	int N = arguments->N;
	double time = (comp_time.tv_sec - start_time.tv_sec) + (comp_time.tv_usec - start_time.tv_usec) * 1e-6;

	printf("Berechnungszeit:    %f s \n", time);
	printf("Speicherbedarf:     %f MiB\n", (N + 2) * (arguments->rows + 2) * sizeof(double) * arguments->num_matrices / 1024.0 / 1024.0);
	printf("Berechnungsmethode: ");

	if (options->method == METH_GAUSS_SEIDEL)
	{
		printf("Gauss-Seidel");
	}
	else if (options->method == METH_JACOBI)
	{
		printf("Jacobi");
	}

	printf("\n");
	printf("Interlines:         %" PRIu64 "\n",options->interlines);
	printf("Stoerfunktion:      ");

	if (options->inf_func == FUNC_F0)
	{
		printf("f(x,y) = 0");
	}
	else if (options->inf_func == FUNC_FPISIN)
	{
		printf("f(x,y) = 2pi^2*sin(pi*x)sin(pi*y)");
	}

	printf("\n");
	printf("Terminierung:       ");

	if (options->termination == TERM_PREC)
	{
		printf("Hinreichende Genaugkeit");
	}
	else if (options->termination == TERM_ITER)
	{
		printf("Anzahl der Iterationen");
	}

	printf("\n");
	printf("Anzahl Iterationen: %" PRIu64 "\n", results->stat_iteration);
	printf("Norm des Fehlers:   %e\n", results->stat_precision);
	printf("\n");
	}
}

/****************************************************************************/
/** Beschreibung der Funktion DisplayMatrix:                               **/
/**                                                                        **/
/** Die Funktion DisplayMatrix gibt eine Matrix                            **/
/** in einer "ubersichtlichen Art und Weise auf die Standardausgabe aus.   **/
/**                                                                        **/
/** Die "Ubersichtlichkeit wird erreicht, indem nur ein Teil der Matrix    **/
/** ausgegeben wird. Aus der Matrix werden die Randzeilen/-spalten sowie   **/
/** sieben Zwischenzeilen ausgegeben.                                      **/
/****************************************************************************/
//Geändert gestellte Variante von Display Matrix verwendet
void DisplayMatrix ( char *s, double *v, int interlines , int rank , int size, int from, int to )
{
  int x,y;
  int lines = 8 * interlines + 9;
  MPI_Status status;

  /* first line belongs to rank 0 */
  if (rank == 0)
    from--;

  /* last line belongs to rank size-1 */
  if (rank + 1 == size)
    to++;

  if (rank == 0)
    printf ( "%s\n", s );

  for ( y = 0; y < 9; y++ )
  {
    int line = y*(interlines+1);

    if (rank == 0)
    {
      /* check whether this line belongs to rank 0 */
      if (line < from || line > to)
      {
        /* use the tag to receive the lines in the correct order
         * the line is stored in v, because we do not need it anymore */
        MPI_Recv(v, lines, MPI_DOUBLE, MPI_ANY_SOURCE, 42 + y, MPI_COMM_WORLD, &status);
      }
    }
    else
    {
      if (line >= from && line <= to)
      {
        /* if the line belongs to this process, send it to rank 0
         * (line - from + 1) is used to calculate the correct local address */
        MPI_Send(&v[(line - from + 1)*lines], lines, MPI_DOUBLE, 0, 42 + y, MPI_COMM_WORLD);
      }
    }

    for ( x = 0; x < 9; x++ )
    {
      if (rank == 0)
      {
        if (line >= from && line <= to)
        {
          /* this line belongs to rank 0 */
          printf ( "%7.4f", v[line*lines + x*(interlines+1)]);
        }
        else
        {
          /* this line belongs to another rank and was received above */
          printf ( "%7.4f", v[x*(interlines+1)]);
        }
      }
    }

    if (rank == 0)
      printf ( "\n" );
  }
  fflush ( stdout );
}

static
void
DisplayMatrixOld (struct calculation_arguments* arguments, struct calculation_results* results, struct options* options)
{
	//TODO DisplayMatrixOld gänzlich entfernen und Aufruf sinnvoll in Main einbauen.
	DisplayMatrix("test", arguments->M, options->interlines , arguments->rank , arguments->num_processes, arguments->firstRow, arguments->firstRow+arguments->rows-1);
}

/* ************************************************************************ */
/*  main                                                                    */
/* ************************************************************************ */
int
main (int argc, char** argv)
{
	int rc = MPI_Init(&argc, &argv);
	if (rc != MPI_SUCCESS)
	{
		fprintf(stderr, "MPI_Init failed!\n");
		exit(EXIT_FAILURE);
	}
	struct options options;
	struct calculation_arguments arguments;
	struct calculation_results results;

	/* get parameters */
	AskParams(&options, argc, argv);              /* ************************* */

	initVariables(&arguments, &results, &options);           /* ******************************************* */

	allocateMatrices(&arguments);        /*  get and initialize variables and matrices  */
	initMatrices(&arguments, &options);            /* ******************************************* */

	gettimeofday(&start_time, NULL);                   /*  start timer         */
	calculate(&arguments, &results, &options);                                      /*  solve the equation  */
	gettimeofday(&comp_time, NULL);                   /*  stop timer          */


	displayStatistics(&arguments, &results, &options);
	DisplayMatrixOld(&arguments, &results, &options);

	freeMatrices(&arguments);                                       /*  free memory     */
	MPI_Finalize();

	return 0;
}
