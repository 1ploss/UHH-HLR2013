#ifndef PI
#define PI 			3.141592653589793
#endif
#define TWO_PI_SQUARE 		(2 * PI * PI)

/****************************************************************************************/
/* Berechnet einen Schritt																*/
/* Argumente:																			*/
/* - matrix_In : 	Pointer auf die Eingabematrix										*/
/* - matrix_Out : 	Pointer auf die Ausgabematrix										*/
/* - out :			Angabe, welcher Index grad als Ausgabe genutzt werden soll			*/
/* - rows :			Angabe darüber, wieviele Spalten die Matrix hat						*/
/* - lines :		Angabe darüber, wieviele Zeilen die Matrix hat						*/
/* - offset :		Der globale Index der ersten Zeile									*/
/* - maxresiduum : 	Der größte Fehler, dieser Iteration									*/
/* - stoer :		1 wenn die Störfunktion verwendet wird, sonst 0						*/
/* - borders :		Pointer auf die Randzeilen, welche aus anderen Prozessen stammen	*/
/* - iteration :	-1, wenn nach Genauigkeit terminiert wird, sonst Iterationen übrig	*/
/****************************************************************************************/
static
void
jacobicalculate (double** matrix_In,double** matrix_Out,int rows,int lines,int offset,double* maxresiduum,int stoer,double** borders, int iteration)
{

	double pih = 0.0;
	double fpisin = 0.0;
	double star = 0.0;

	if(stoer)
	{
		pih = PI * (1.0/(double)rows);
		fpisin = 2 * TWO_PI_SQUARE;
	}
	*maxresiduum= 0.0;
	
	for (int i = 1; i< rows;i++)
	{	
		double fpisin_i = 0.0;
		
		if(stoer)
		{
			fpisin_i = fpisin * sin(pih * (double)i);
		}
		for(int j = 0; i< lines; j++)
		{
			if(j==0)
			{
				star = 0.25 * (matrix_In[i-1][j] + borders[i][0] + matrix_In[i][j+1] + matrix_In[i+1][j]);
			}else if(j==lines -1)
			{
				star = 0.25 * (matrix_In[i-1][j] + matrix_In[i][j-1] + borders[i][1] + matrix_In[i+1][j]);
			}else
			{
				star = 0.25 * (matrix_In[i-1][j] + matrix_In[i][j-1] + matrix_In[i][j+1] + matrix_In[i+1][j]);
			}

			if(stoer)
			{
				star += fpisin_i * sin(pih * (double)j);
			}

			if(iteration >-1)
			{
				double residuum = matrix_In[i][j] - star;
				residuum = (residuum < 0) ? -residuum : residuum;
				*maxresiduum = (residuum < *maxresiduum) ? *maxresiduum : residuum;
			}

			matrix_Out[i][j] = star;
		}
	}
}

static
void
initMatrix(int N, int pnr, int lines, int stoer,double*** matrizen)
{
	double h = 1.0/(double)N;
	int globali =0;

	matrizen = malloc(2*sizeof(double**));

	for(int n =0;n<2;n++)
	{
		matrizen[n] = calloc(lines,sizeof(double*));

		for(int i =0;i<lines;i++)
		{
			matrizen[n][i] = calloc(N,sizeof(double));
		}
	}
	if(!stoer)
	{
		for(int n =0;n<2;n++)
			{
				for(int i =0;i<lines;i++)
				{
					globali = (i+(pnr*lines));
					matrizen[n][i][0] = 1.0 - (h * globali);
					matrizen[n][i][N] = h * globali;
					matrizen[n][0][i] = 1.0 - (h * globali);
					matrizen[n][N][i] = h * globali;
				}
				matrizen[n][N][0] = 0.0;
				matrizen[n][0][N] = 0.0;
			}
	}
}

static
void
freeMatrizen(double***matrizen,int lines)
{
	for(int n = 0;n<2;n++)
	{
		for(int i = 0; i<lines;i++)
		{
			free(matrizen[n][i]);
		}
		free(matrizen[n]);
	}
	free(matrizen);
}


int main(int argc, char **argv) {
	int interlines;
	int terminationCondition;
	int N = (interlines * 8) + 9;

	if(argc<3)
	{
		printf("Zu wenig Argumente");
	}
}














