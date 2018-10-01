# -*- coding: utf-8 -*-
import sys
import time
import matplotlib.pyplot as plt
import random
import pickle
import queue
import numpy as np
from sklearn.linear_model import LinearRegression
sys.path.append(r"D:\practicas_DAA_2017")
import grafos as gr

#   Función que devuelve una matriz de adyacencia de un grafo ponderado (con pesos)
#   con n_nodes, una proporción sparse_factor de ramas (?¿) y max_weight como peso máximo
def rand_matr_pos_graph(n_nodes, sparse_factor, max_weight=50., decimals=0):    ## TODO: Usar el decimals

    matBinaria = np.random.binomial(1, sparse_factor, (n_nodes, n_nodes))           # Creamos la matriz que contiene las conexiones
    matPesos = np.random.binomial(max_weight, sparse_factor, (n_nodes, n_nodes))    # Creamos la matriz que contiene los pesos
    matAdyacencia = np.empty((n_nodes, n_nodes))                                    # Creamos la matriz de adyacencia
    matAdyacencia.fill(np.inf)                                                      # La rellenamos de infinito

    matFinal = matBinaria * matPesos            # Multiplicamos la primera matriz por la segunda para tener los pesos SOLO en las conexiones generadas

    for i in range(matAdyacencia.shape[0]):     # Recorremos la matriz de adyacencia
        for j in range(matAdyacencia.shape[1]):
            if i == j:                                  # Para poner a 0 la diagonal
                matAdyacencia[i][j] = 0
            elif matFinal[i][j] != 0:                   # Y para rellenar las conexiones
                matAdyacencia[i][j] = matFinal[i][j]

    return matAdyacencia

#   Funcion que devuelve el numero de ramas en el grafo dada una matriz de adyacencia
def cuenta_ramas(m_g):

    num_ramas = 0

    for filas in m_g:                           # Bucle que recorre la matriz de adyacencia
        for peso in filas:
            if peso != 0 and peso != np.inf:            # Sumando 1 cuando encuentra una conexión
                num_ramas += 1

    return num_ramas

#   Función que genera las matrices de n_grafos aleatorios con n_nodes y un cierto
#   sparse_factor, devolviendo la media de sparse_factor reales de las matrices generadas
def check_sparse_factor(n_grafos, n_nodes, sparse_factor):

    sp_aux = 0          # Inicializamos a 0 esta variable auxiliar

    for i in range(n_grafos):       # Hacemos un bucle tantas veces como grafos tengamos que generar
        mat = rand_matr_pos_graph(n_nodes, sparse_factor)   # Generamos la matriz de adyacencia
        sp = cuenta_ramas(mat)                              # Contamos las ramas de la matriz generada
        sp_aux += sp                                        # Sumamos las ramas al contador

    avg_sparse_factor = sp_aux/n_grafos     # Por ultimo, calculamos el factor de dispersion medio

    return avg_sparse_factor

#   Función que devuelve el diccionario de listas de adyacencia
#   del grafo decinido por la matriz de adyacencia m_g
#   SE SUPONE QUE TIENE QUE GENERAR UN DICCIONARIO DE UNA MATRIZ (?¿)
def m_g_2_d_g(m_g):

    d_g = {}    # Diccionario a devolver
    index = 0   # Indice para el diccionario G (indice de nodo)

    for nodo in m_g:        # Bucle que recorre la matriz

        d_g[index] = {}     # Diccionario G[i]
        j = 0       # Indice para el diccionario G[i] (no. de nodo)

        for peso in nodo:                       # Recorremos los valores de cada fila (nodo)
            if peso != 0 and peso != np.inf:        # Si tiene peso, lo añadimos al diccionario
                d_g[index][j] = peso                    # j porque pienso que los nodos van de 0-inf, no de 1-inf
            j+=1

        index += 1          # Añadimos 1 al indice del diccionario G

    return d_g

#   Función que devuelve una matriz de adyacencia
#   dado un diccionario de listas de adyacencia d_g
def d_g_2_m_g(d_g):

    m_g = np.empty((len(d_g), len(d_g)))    # Como son matrices cuadradas, la longitud del diccionario nos da la cantidad de nodos
    m_g.fill(np.inf)                        # La rellenamos de np.inf

    for i in range(len(d_g)):               # Bucle para inicializar la diagonal a 0
        m_g[i][i] = 0

    for k1, v1 in d_g.items():              # Recorremos el diccionario. k1 = indice ; v1 = dic. de ese nodo
        for k2, v2 in v1.items():               # Recorremos el diccionario del nodo
            m_g[k1][k2] = v2                        # Colocamos el peso en la posición que le corresponde

    return m_g

#   Función que guarda un objeto Python obj de manera comprimida en un fichero de nombre f_name
def save_object(obj, f_name='obj.pklz', save_path='.'):

    objFile = open(save_path + f_name, 'wb')    # Abrimos el fichero en modo de escritura binaria para que funcione pickle
    pickle.dump(obj, objFile)                   # Guardamos el objeto ¿ya serializado? en el fichero
    objFile.close()                             # Cerramos el fichero

#   Función que devuelve un objeto Python guardado en un fichero de nombre f_name
def read_object(f_name, save_path='.'):

    objFile = open(save_path + f_name, 'rb')    # Abrimos el fichero en modo de lectura binaria para que funcione pickle
    object = pickle.load(fp)                    # Cargamos el objeto ¿serializado? guardado en el fichero
    objFile.close()                             # Cerramos el fichero

    return object

#   Función que escribe en un fichero de nombre f_name un grafo ponderado en formato
#   TGF a partir de un diccionario de listas de adyacencia
def d_g_2_TGF(d_g, f_name):

    TGFFile = open(f_name, 'w')                 # Abrimos el fichero en modo escritura

    for indice in d_g:                          # Escribimos primero los indices en el fichero
        TGFFile.write(indice + '\n')

    TGFFile.write('#\n')                        # Escribimos el separador

    for nodoOrg, DiccDestinos in d_g:           # Recorremos de nuevo el diccionario para escribir
        for nodoDst, pesoRec in DiccDestinos:       # el resto del fichero
            TGFFile.write(nodoOrg + ' ' + nodoDst + ' ' + pesoRec + '\n')   # Escribimos los diferentes datos

    TGFFile.close()                             # Cerramos el fichero

#   Función que devuelve un diccionario de listas de adyacencia a partir de un
#   grafo ponderado TGF guardado en el archivo f_name   ## TODO: Se guarda como str en el dicc
def TGF_2_d_g(f_name):

    d_g = {}                                    # Inicializamos el diccionario de listas de adyacencia

    TGFFile = open(f_name, 'r')                 # Abrimos nuestro fichero donde tenemos el grafo en TGF

    for linea in TGFFile:                       # Leemos cada linea del fichero para obtener los indices de los diferentes nodos
        indice = linea.split('\n')[0]               # Dividimos la linea y cogemos el indice del nodo
        if indice == '#':                           # Cuando leemos el '#' salimos del bucle ya que hemos leido todos los indices
            break
        else:                                       # Si no leemos '#', guardamos el indice y creamos un diccionario para el mismo
            d_g[indice] = {}

    for linea in TGFFile:                       # Ahora guardamos en los distintos diccionarios los nodos destino y los costes hacia ellos
        nodoOrg, nodoDst, pesoRec = linea.split('\n')[0].split(' ')    # Fragmentamos la linea leida en los diferentes datos
        d_g[nodoOrg][nodoDst] = pesoRec                                # Creamos una clave y valor nuevos para el diccionario de nodoOrg

    TGFFile.close()                             # Cerramos el fichero

    return d_g

def dijkstra_d(d_g, u):

    lstOpenNodes = [] #s
    d_prev = {} #p
    d_dist = {} #d

    for i in range(len(d_g)):
        lstOpenNodes.append(False)

    Q = queue.PriorityQueue()

    d_dist[u] = 0 # K : nodoDst ; V : peso
    Q.put((d_dist[u],u)) # [0]: Peso; [1]: NodoDst

    # Mientras la cola de prioridad no este vacia
    while not Q.empty():

        dist_tuple = Q.get()
        dist = dist_tuple[0]
        nodoActual = dist_tuple[1]

        if not lstOpenNodes[nodoActual]:

            d_dist[nodoActual] = dist
            lstOpenNodes[nodoActual] = True
            # Sacamos las adyacencias del nodo a analizar:
            diccionario_adyacencias = d_g[nodoActual]
            # Sacamos las conexiones
            for dicc_NodeDst, dicc_Dist in diccionario_adyacencias.items():

                #   Si no existe, lo creamos
                if not dicc_NodeDst in d_dist:
                    d_dist[dicc_NodeDst] = dicc_Dist + dist
                    d_prev[dicc_NodeDst] = nodoActual
                    Q.put((d_dist[dicc_NodeDst],dicc_NodeDst))

                #   Si existe, comprobamos si tiene menor peso que el anterior guardado
                elif d_dist[dicc_NodeDst] > (d_dist[nodoActual] + d_g[nodoActual][dicc_NodeDst]):
                    d_dist[dicc_NodeDst] = d_dist[nodoActual] + d_g[nodoActual][dicc_NodeDst]
                    d_prev[dicc_NodeDst] = nodoActual
                    Q.put((d_dist[dicc_NodeDst],dicc_NodeDst))
