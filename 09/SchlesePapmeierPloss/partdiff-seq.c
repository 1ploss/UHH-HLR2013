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

struct calculation_arguments
{
	uint64_t  N;              /* number of spaces between lines (lines=N+1)     */
	uint64_t  num_matrices;   /* number of matrices                             */
	double    h;              /* length of a space between two lines            */
	double    ***Matrix;      /* index matrix used for addressing M             */
	double    *M;             /* two matrices with real values                  */
	uint64_t  firstRow;		  /* global Index of the local Index 0				*/
	uint64_t  rows;			  /* Number of local rows							*/
	int		  rank;		   	  /* Rank of local process							*/
	int		  num_processes;  /* Number of used processes						*/
	//TODO möglicherweise müssen wir hier noch mehr einführen
	//Rank eingef�hrt, weil man den eh �fter mal braucht und dann gleich �ber initVariables gesetzt werden kann.
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
	//TODO Initialisierung anpassen
	arguments->N = (options->interlines * 8) + 9 - 1;
	arguments->num_matrices = (options->method == METH_JACOBI) ? 2 : 1;
	arguments->h = 1.0 / arguments->N;

	/*Berechne die Anzahl an Zeilen pro Prozess. Zeilen nur die zu berechnenden Zeilen hier.
	 * obere extra Zeile und untere extra Zeile m�ssen extra beachtet werden?*/
	int rank, num_tasks;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);
	div_t d = div(arguments->N+1, num_tasks);   //N ist Indexbehaftet also f�r Matrixzugriff von 0 bis (echte Zeilen-1)
	unsigned numLines = d.quot;  //Jeder Prozess zun�chst GesamtZeilen/ProzessAnzahl abgerundet
	/*Berechne schonmal die first rows, da diese nur Bezug auf die echte Matrix haben und die zus�tzlichen 2 Zeilen nicht beachten m�ssen.
	 * Achtung firstRow hier interpretiert als erste echte Matrixzeile, die der Prozess auch selber berechnet. Daf�r die Angabe der Zeile in der gesamten Matrix*/
	unsigned firstRow = rank * d.quot; //So werden die zus�tzlichen Zeilen der n Prozesse rank < d.rem noch nicht beachtet
	if(rank < d.rem)
	{
		numLines++;    //Wenn lokal rank kleiner als Rest extra Zeile
		firstRow = firstRow + rank;    /*firstRow um anzahl vorheriger Prozesse mit extra Zeile erh�hen.
											Entspricht immer genau eigenem Rang*/
	}
	else
	{
		firstRow = firstRow + d.rem;   /*Wenn Rang h�her als Rest gab es bereits N=Rest Prozesse mit extra Zeile*/
	}
	if(rank == 0) //Gibt so die erste wirklich zu berechnende Zeile und Anzahl Zeilen an. Nimmt sp�ter einige Probleme
	{
		firstRow = 1;
		numLines--;
	}
	if(rank == num_tasks-1)   //�hnlich nur das firstRow stimmt aber auch hier eine Zeile weniger berechnet wird(wegen R�ndern der Matrix)
	{
		numLines--;
	}
	arguments->firstRow = firstRow;
	arguments->rows = numLines;
	arguments->rank = rank;
	arguments->num_processes = num_tasks;
	/* Diese Berechnung f�r die firstRow und rows an ein paar kleinen Beispielen getestet. Scheint zu funktionieren.
	printf("num_lines: %d \n", numLines);
	printf("FirstRow: %d\n",arguments->firstRow);
	printf("rows: %d\n",arguments->rows);
	*/

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
	//TODO anpassen
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

	//TODO anpassen
	arguments->M = allocateMemory(arguments->num_matrices * (N + 2) * (rows + 2) * sizeof(double));  //Hoffentlich stimmt das mit N+2 Die gesamte Matrix ist so 2 Spalten breiter. Ziemlicher Overkill
																									//funktioniert wohl auch nicht so wie ich das dachte mit dem residuum ans Ende setzen
	arguments->Matrix = allocateMemory(arguments->num_matrices * sizeof(double**));

	for (i = 0; i < arguments->num_matrices; i++)
	{
		arguments->Matrix[i] = allocateMemory((N + 1) * sizeof(double*));

		for (j = 0; j <= N; j++)
		{
			arguments->Matrix[i][j] = arguments->M + (i * (N + 1) * (N + 1)) + (j * (N + 1));
		}
	}

	/*Orignal Code
	uint64_t i, j;

	uint64_t const N = arguments->N;

	//TODO anpassen
	arguments->M = allocateMemory(arguments->num_matrices * (N + 1) * (N + 1) * sizeof(double));
	arguments->Matrix = allocateMemory(arguments->num_matrices * sizeof(double**));

	for (i = 0; i < arguments->num_matrices; i++)
	{
		arguments->Matrix[i] = allocateMemory((N + 1) * sizeof(double*));

		for (j = 0; j <= N; j++)
		{
			arguments->Matrix[i][j] = arguments->M + (i * (N + 1) * (N + 1)) + (j * (N + 1));
		}
	}
	*/
}

/* ************************************************************************ */
/* initMatrices: Initialize matrix/matrices and some global variables       */
/* ************************************************************************ */
static
void
initMatrices (struct calculation_arguments* arguments, struct options const* options)
{
	uint64_t g, i, j,globali;                                /*  local variables for loops   */

	uint64_t const N = arguments->N;
	double const h = arguments->h;
	double*** Matrix = arguments->Matrix;
	uint64_t const rows = arguments->rows;
	uint64_t const firstRow = arguments->firstRow;
	/* initialize matrix/matrices with zeros */
	for (g = 0; g < arguments->num_matrices; g++)
	{
		//Weglassen und bei allocate calloc verwenden?
		for (i = 0; i < rows+2; i++)//TODO überprüfen, ob das so stimmt.
									//Jetzt sind erster und letzter Rank nicht gesondert betrachtet
									//So m�sste die Variante sein, wenn wir die gesamte verwendete Matrix mit 0 initialisieren wollen
									//wobei rows als Zeilen die wirklich berechnet werden angesehen ist
									//Funktioniert so auch
		{
			for (j = 0; j <= N; j++)
			{
				Matrix[g][i][j] = 0.0;
			}
		}
		/* Original Variante
		 for (i = 0; i < N; i++)//TODO überprüfen, ob das so stimmt
		{
			for (j = 0; j <= N; j++)
			{
				Matrix[g][i][j] = 0.0;
			}
		}*/
	}

	/* initialize borders, depending on function (function 2: nothing to do) */
	if (options->inf_func == FUNC_F0)   //Ab hier sollte alles korrekt funktionieren unter den Annahmen,
										//dass alle Prozesse 2 extra Zeilen mit initialisieren, die Nachbarprozessen geh�ren
										//au�er erster und letzter Prozess die nur 1 extra Zeile von Nachbarprozessen haben aber daf�r 1 RandZeile besitzen
	{
		for (g = 0; g < arguments->num_matrices; g++)
		{

				for (i = 0; i < rows+2; i++)
				{
					globali=firstRow+i-1;   //Entschieden f�r erster und letzter Rank haben eine extra Zeile f�r Nachbarprozesse weniger sehen aber Randzeile als extra Zeile an
					Matrix[g][i][0] = 1.0 - (h * globali);
					Matrix[g][i][N] = h * globali;
				}
			if(arguments->rank==0||arguments->rank==(arguments->num_processes-1))
			{
				for(i = 0; i <= N; i++)
				{
					if(arguments->rank==0)
					{
						Matrix[g][0][i] = 1.0 - (h * i);
					}
					else
					{
						Matrix[g][rows+1][i] = h * i;  //Beachte N ist Indexbehaftet. Sollte jetzt denke ich alles richtig sein
					}
				}
			}
			if(arguments->rank==(arguments->num_processes-1))
			{
				Matrix[g][rows+1][0] = 0.0;
			}
			if(arguments->rank==0)
			{
				Matrix[g][0][N] = 0.0;
			}

			/* Original Variante
			for (i = 0; i <= N; i++)
			{
				Matrix[g][i][0] = 1.0 - (h * i);
				Matrix[g][i][N] = h * i;
				Matrix[g][0][i] = 1.0 - (h * i);
				Matrix[g][N][i] = h * i;
			}

		Matrix[g][N][0] = 0.0;
		Matrix[g][0][N] = 0.0;
		*/
		}
	}
}

/* ************************************************************************ */
/* calculate: solves the equation                                           */
/* ************************************************************************ */
static
void
calculate (struct calculation_arguments const* arguments, struct calculation_results *results, struct options const* options)
{
	int i, j, globali;                          /* local variables for loops  */
	int m1, m2;                                 /* used as indices for old and new matrices       */
	double star;                                /* four times center value minus 4 neigh.b values */
	double residuum;                            /* residuum of current iteration                  */
	double maxresiduum;                         /* maximum residuum value of a slave in iteration */

	int const N = arguments->N;
	double const h = arguments->h;
	int const rows = arguments->rows;    //TODO extra Variablen �berpr�fen
	int const rank = arguments->rank;
	int const firstrow = arguments->firstRow;
	int const lastRank = arguments->num_processes-1;
	int const firstRank = 0;
	MPI_Request sendRequest, recvRequest;
	MPI_Status status;

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
	//TODO calc umbauen

	if (options->inf_func == FUNC_FPISIN)
	{
		pih = PI * h;
		fpisin = 0.25 * TWO_PI_SQUARE * h * h;
	}

	while (term_iteration > 0)
	{
		double** Matrix_Out = arguments->Matrix[m1];
		double** Matrix_In  = arguments->Matrix[m2];

		maxresiduum = 0;

		//Bisher nur f�r Gauss-Seidel geschrieben. Sollte noch ne Berechnungsmethodenabfrage bekommen
		//Ist das hier so richtig, dass es nur 1 mal pro Iteration ausgef�hrt wird?
		if(rank>firstRank)
		{
			MPI_Recv(Matrix_Out[0],N+3,MPI_DOUBLE, rank-1,0,MPI_COMM_WORLD, &status); //Tags nur sehr provisorisch verwendet zum Testen
		}
		if(!(results->stat_iteration == 0) & (rank<lastRank)) //Erhalten der 1. Zeile des unteren Prozesses kann jederzeit w�hrend berechnung geschehen.
		{
			MPI_Irecv(Matrix_Out[rows+1],N+3,MPI_DOUBLE, rank+1,1,MPI_COMM_WORLD, &recvRequest);  //Hier Irecv zum Segfault (behoben)
		}

		/* over all rows */
		for (i = 1; i <= rows; i++)    //TODO rows richtig einsetzen Done funktioniert so bisher richtig
		{
			double fpisin_i = 0.0;
			if((rank>firstRank) & (i == 2)) //Im 2 Durchlauf ist die 1. Zeile berechnet und kann nach oben weiter gegeben werden. Muss aber nur 1 Mal gemacht werden
			{
				MPI_Isend(Matrix_Out[1],N+3,MPI_DOUBLE, rank-1, 1, MPI_COMM_WORLD, &sendRequest);  //Hier Isend zum Segfault (behoben)
			}

			if (options->inf_func == FUNC_FPISIN)
			{
				globali = firstrow+i-1;
				fpisin_i = fpisin * sin(pih * (double)globali);    //TODO globali richtig einsetzen (Done) nicht sicher aber wahrscheinlich richtig
																	//Berechnete Werte waren lokal im Vergleich zur sequentiellen Variante gleich
			}
			if((i==rows) & (rank<lastRank) & !(results->stat_iteration == 0)) //Seqfault lag daran, dass &recvRequest erst ab stat_iteration > 0 benutzt wird
			{
				MPI_Wait(&recvRequest, MPI_STATUS_IGNORE);  //Produziert Segfault, wieso?
			}

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
					if(rank>firstRank)
					{
					//	maxresiduum = (Matrix_Out[0][N+1] < maxresiduum) ? maxresiduum : Matrix_Out[0][N+1]; //Einbeziehen des erhaltenen maxresiduum des vorherigen Prozess
					}
					if((i==rows) & (j==(N-1)))               //Braucht nur einmal am Ende der Iteration geschrieben werden und nur in der letzten Iteration (wenn nach Iterationen abgebrochen wird)
					{
					//	Matrix_Out[rows][N+1] = maxresiduum; //Speichert das MaxResiduum in der letzten Zeile und in einer extra Spalte
															 //Um nur ein Send einer Zeile ausf�hren zu m�ssen
															//TODO maxresiduum kann so nicht einfach mit gesendet werden, da bei der fortlaufenden Adressierung wieder Adresse der n�chsten Zeile an Stelle 0 erreicht wird
					}
				}

				Matrix_Out[i][j] = star;
			}
		}
		//Nach jedem Durchlauf die letzte Zeile nach unten senden
		if(rank<lastRank)
		{
			MPI_Send(Matrix_Out[rows], N+3, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD);
		}

		results->stat_iteration++;
		results->stat_precision = maxresiduum;

		/* exchange m1 and m2 */
		i = m1;
		m1 = m2;
		m2 = i;

		/* check for stopping calculation, depending on termination method */
		if (options->termination == TERM_PREC)  //TODO �ndern
		{
			if (maxresiduum < options->term_precision)
			{
				term_iteration = 0;
			}
		}
		else if (options->termination == TERM_ITER) //M�sste schon so stimmen
		{
			term_iteration--;
		}
	}

	results->m = m2;
}

/* ************************************************************************ */
/*  displayStatistics: displays some statistics about the calculation       */
/* ************************************************************************ */
static
void
displayStatistics (struct calculation_arguments const* arguments, struct calculation_results const* results, struct options const* options)
{
	if(arguments->rank==0)
	{
	int N = arguments->N;
	double time = (comp_time.tv_sec - start_time.tv_sec) + (comp_time.tv_usec - start_time.tv_usec) * 1e-6;

	printf("Berechnungszeit:    %f s \n", time);
	printf("Speicherbedarf:     %f MiB\n", (N + 3) * (arguments->rows + 2) * sizeof(double) * arguments->num_matrices / 1024.0 / 1024.0);
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
	//TODO DisplayMatrixOld g�nzlich entfernen und Aufruf sinnvoll in Main einbauen.
	DisplayMatrix("test", arguments->M, options->interlines , arguments->rank , arguments->num_processes, arguments->firstRow, arguments->firstRow+arguments->rows-1);
	/*if(arguments->rank==2)
	{
	int x, y;

	double** Matrix = arguments->Matrix[results->m];

	int const interlines = options->interlines;

	printf("Matrix:\n");

	for (y = 0; y < 9; y++)
	{
		for (x = 0; x < 9; x++)
		{
			printf ("%7.4f", Matrix[y * (interlines + 1)][x * (interlines + 1)]);
		}

		printf ("\n");
	}

	fflush (stdout);
	}*/
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


	displayStatistics(&arguments, &results, &options);//TODO dies nur von einem Prozess ausführen lassen
	DisplayMatrixOld(&arguments, &results, &options);//TODO hier möglichst die paralele Version einfügen

	freeMatrices(&arguments);                                       /*  free memory     */
	MPI_Finalize();

	return 0;
}