#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""

    Módulo que trabaja con las listas de adyacencia de adyacencia de los grafos para
    resolver problemas propios de los grafos así como el algoritmo de Dijkstra y el
    calculo de la complejidad de computación del mismo.

    En este modulo se trabaja con 3 diferentes tipos de formas de expresar la matriz
    de adyacencia de un grafo, siendo estos:

        +   Matriz
                Array bidimensional creado por la libreria de Python Numpy, las cuales
                se expresan de la siguiente manera:

                    [[ 0. inf inf inf inf]
                     [inf  0. inf 15. inf]
                     [inf inf  0. inf  7.]
                     [10. inf 14.  0. 11.]
                     [inf inf inf inf  0.]]

                Donde "inf" son distancias infinitas y por tanto que NO hay conexión
                entre esos dos nodos, y las demás distancias son los costes que están
                expresadas en tipo float.

        +   Diccionario
                Diccionario de diccionario de diccionarios los cuales se expresan de
                la siguiente manera:

                {0: {}, 1: {3: 15.0}, 2: {4: 7.0}, 3: {0: 10.0, 2: 14.0, 4: 11.0}, 4: {}}

                Donde la primera clave (K1) es el indice de un nodo y su valor es otro
                diccionario, donde las claves de este son los nodos a los que está
                conectado K1 y el valor es el peso de la rama entre ellos.

        +   TGF
                El formato TGF sirve para guardar un grafo junto a sus ramas en un fichero
                para que se pueda guardar y serializar. Su formato es el siguiente:

                0
                1
                2
                3
                4
                #
                1 3 15.
                2 4 7.
                3 0 10.
                3 2 14.
                3 4 11.

                Donde la primera parte (la que va antes del #) nos muestra los diferentes
                indices de un grafo, y la segunda parte, nos indica que nodo esta conectado
                con quien y el peso de esa rama.
                
        +   Grafo dirigido NetworkX
                
                La estructura usa un diccionario de diccionarios de diccionarios para guardar 
                un grafo. El primer diccionario d tiene como claves [u] los indices de los 
                diferentes nodos del grafo. Los valores contenidos en estos, d[u], son diccionarios
                donde las claves son todos los diferentes nodos v a los que el nodo u esta conectado.
                Finalmente, cada nodo destino v, tiene como valor un diccionario, los cuales contienen
                como clave un atributo del nodo, sea este el peso, color, edad... y valor el mismo valor
                del atributo. La estructura de un grafo dirigido NetworkX es el siguiente:
                
                {0: {2: {'weight': 27.0}, 3: {'weight': 25.0}}, 2: {0: {'weight': 24.0}}, 
                3: {0: {'weight': 24.0}, 2: {'weight': 15.0}}, 1: {4: {'weight': 22.0}}, 
                4: {2: {'weight': 29.0}, 3: {'weight': 29.0}}}
                
                Por tanto:
                
                    {u: {v: {'weight': p}, (...)}. (...)}
                    
                    u --------------------> Nodo Origen
                    v --------------------> Nodo Destino
                    d[u] -----------------> v
                    d[u][v] --------------> Atributos de v
                    d[u][v]['weight'] ----> Peso de la rama (u v)
                    
    @author: Carlos Gonzalez García & Andrés Calvente Rodríguez
"""


# In[ ]:


# -*- coding: utf-8 -*-
import sys
import time
import matplotlib.pyplot as plt
import random
import pickle
import queue
import numpy as np
import networkx as nx
from sklearn.linear_model import LinearRegression
import pandas as pd
#sys.path.append(r'D:\GoogleDriveBck\Cursos\DAA\practicas\2018_2019\python')
#import grafos_2018 as gr


# In[ ]:


def rand_matr_pos_graph(n_nodes, sparse_factor, max_weight=50., decimals=0):    ## TODO: Usar el decimals
    """
    Función que devuelve una matriz de adyacencia de un grafo ponderado (con pesos).

    Parámetros:
    n_nodes ----------> Número de nodos del grafo
    sparse_factor ----> Factor de ramificación del grafo
    max_weight -------> Peso maximo de una ramas
    decimals ---------> Número de decimales de los pesos de las ramas

    Retorno:
    Matriz Numpy de adyacencia del grafo creado.
    """

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


# In[ ]:


grafoMatriz = rand_matr_pos_graph(5,0.5)
print("Matriz Numpy:\n", grafoMatriz)


# In[ ]:


def cuenta_ramas(m_g):
    """
    Funcion que devuelve el numero de ramas en el grafo dada una matriz de adyacencia.

    Parámetros:
    m_g --> Grafo en formato Matriz Numpy

    Retorno:
    Número de ramas del grafo.
    """

    num_ramas = 0

    for filas in m_g:                           # Bucle que recorre la matriz de adyacencia
        for peso in filas:
            if peso != 0 and peso != np.inf:            # Sumando 1 cuando encuentra una conexión
                num_ramas += 1

    return num_ramas


# In[ ]:


numRamas = cuenta_ramas(grafoMatriz)
print("Numero de ramas: ", numRamas)


# In[ ]:


def check_sparse_factor(n_grafos, n_nodes, sparse_factor):
    """
    Función que genera un número de grafos aleatorios y calcula el factor d
    ramificación medio sobre todos los grafos generados.

    Parámetros:
    n_grafos ---------> Número de grafos a generar
    n_nodes ----------> Número de nodos por grafo
    sparse_factor ----> Factor de ramificación de los grafos

    Retorno:
    Factor de ramificación medio.
    """

    sp_aux = 0          # Inicializamos a 0 esta variable auxiliar

    for i in range(n_grafos):       # Hacemos un bucle tantas veces como grafos tengamos que generar
        mat = rand_matr_pos_graph(n_nodes, sparse_factor)   # Generamos la matriz de adyacencia
        sp = cuenta_ramas(mat)                              # Contamos las ramas de la matriz generada
        sp_aux += sp                                        # Sumamos las ramas al contador

    avg_sparse_factor = sp_aux/n_grafos     # Por ultimo, calculamos el factor de dispersion medio

    return avg_sparse_factor


# In[ ]:


sparseFactorMat = check_sparse_factor(10,5,0.5)
print("Sparse Factor: ", sparseFactorMat)


# In[ ]:


def m_g_2_d_g(m_g):
    """
    Función que pasa una matriz de adyacencia en formato Matriz Numpy a formato Diccionario.

    Parámetros:
    m_g --> Matriz Numpy de adyacencia a transformar

    Retorno:
    El diccionario resultante a partir de la matriz.
    """

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


# In[ ]:


grafoDicc = m_g_2_d_g(grafoMatriz)
print("Matriz Numpy convertida a Diccionario:\n", grafoDicc)


# In[ ]:


def d_g_2_m_g(d_g):
    """
    Función que pasa una matriz adyacencia en formato Diccionario a formato Matriz Numpy.

    Parámetros:
    d_g --> Diccionario de una matriz de adyacencia a transformar

    Retorno:
    La Matriz Numpy resultante a partir del Diccionario.
    """

    m_g = np.empty((len(d_g), len(d_g)))    # Como son matrices cuadradas, la longitud del diccionario nos da la cantidad de nodos
    m_g.fill(np.inf)                        # La rellenamos de np.inf

    for i in range(len(d_g)):               # Bucle para inicializar la diagonal a 0
        m_g[i][i] = 0

    for k1, v1 in d_g.items():              # Recorremos el diccionario. k1 = indice ; v1 = dic. de ese nodo
        for k2, v2 in v1.items():               # Recorremos el diccionario del nodo
            m_g[k1][k2] = v2                        # Colocamos el peso en la posición que le corresponde

    return m_g


# In[ ]:


print("Matriz Numpy Original:\n", grafoMatriz, "\n")
grafoMatriz=d_g_2_m_g(grafoDicc)
print("Diccionario convertida a Matriz Numpy:\n", grafoMatriz)


# In[ ]:


def save_object(obj, f_name='obj.pklz', save_path='.'):
    """
    Función que guarda un objeto Python de manera comprimida en un fichero.

    Parámetros:
    obj ----------> Objeto Python a comprimir
    f_name -------> Nombre del fichero donde queremos guardar el objeto
    save_path ----> Ruta donde queremos guardar el fichero
    """

    objFile = open(save_path + f_name, 'wb')    # Abrimos el fichero en modo de escritura binaria para que funcione pickle
    pickle.dump(obj, objFile)                   # Guardamos el objeto ¿ya serializado? en el fichero
    objFile.close()                             # Cerramos el fichero


# In[ ]:


def read_object(f_name, save_path='.'):
    """
    Función que carga un objeto Python de un fichero.

    Parámetros:
    f_name -------> Nombre del fichero donde tenemos el objeto
    save_path ----> Ruta donde tenemos el fichero

    Retorno:
    El objeto Python de dentro del fichero
    """

    objFile = open(save_path + f_name, 'rb')    # Abrimos el fichero en modo de lectura binaria para que funcione pickle
    object = pickle.load(fp)                    # Cargamos el objeto ¿serializado? guardado en el fichero
    objFile.close()                             # Cerramos el fichero

    return object


# In[ ]:


def d_g_2_TGF(d_g, f_name):
    """
    Función que escribe en un fichero un grafo ponderado en formato
    TGF a partir de un diccionario de listas de adyacencia.

    Parámetros:
    d_g ------> Diccionario de listas de adyacencia a transformar
    f_name ---> Nombre del fichero queremos guardar el grafo en formato TGF
    """

    TGFFile = open(f_name, 'w')                 # Abrimos el fichero en modo escritura

    for indice in d_g:                          # Escribimos primero los indices en el fichero
        TGFFile.write(indice + '\n')

    TGFFile.write('#\n')                        # Escribimos el separador

    for nodoOrg, DiccDestinos in d_g:           # Recorremos de nuevo el diccionario para escribir
        for nodoDst, pesoRec in DiccDestinos:       # el resto del fichero
            TGFFile.write(nodoOrg + ' ' + nodoDst + ' ' + pesoRec + '\n')   # Escribimos los diferentes datos

    TGFFile.close() 


# In[ ]:


def TGF_2_d_g(f_name):
    """
    Función que lee de un fichero un grafo en formato TGF y crea un diccionario
    de listas de adyacencia a partir del TGF.

    Parámetros:
    f_name --> Nombre del fichero donde tenemos el grafo en TGF

    Retorno:
    Diccionario de listas de adyacencia generado.
    """

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


# In[ ]:


def dijkstra_d(d_g, u):
    """
    Funcion que resuelve el algoritmo de Dijkstra para un nodo de un grafo almacenado
    en un diccionario de listas de adyacencia.

    Parámetros:
    d_g --> Diccionario de listas de adyacencia de un grafo a aplicar Dijkstra sobre uno de sus nodos
    u ----> Nodo del grafo a aplicar Dijkstra

    Retorno:
    +   Diccionario de distancia de u a K con peso V.
    +   Diccionario de previos donde V es el nodo previo a K.
    """

    lstOpenNodes = [] #s    # Lista de nodos abiertos
    d_dist = {} #d          # Diccionario de costes minimos
    d_prev = {} #p          # Diccionario de nodos previos

    for i in range(len(d_g)):       # Inicializamos la lista entera a False
        lstOpenNodes.append(False)  ## TODO: ¿Se puede hacer mejor?

    Q = queue.PriorityQueue()       # Inicializamos la cola de prioridad

    d_dist[u] = 0           # K : nodoDst ; V : peso
    Q.put((d_dist[u],u))    # [0]: Peso; [1]: NodoDst   # Debe ser asi para que funcione la prioridad

    # Mientras la cola de prioridad no este vacia
    while not Q.empty():

        dist_tuple = Q.get()        # Cogemos la tupla (peso, distancia)
        dist = dist_tuple[0]
        nodoActual = dist_tuple[1]

        #   Si el nodo no esta cerrado
        if not lstOpenNodes[nodoActual]:

            lstOpenNodes[nodoActual] = True             # Cerramos el nodo
            diccionario_adyacencias = d_g[nodoActual]   # Sacamos las adyacencias del nodo a analizar

            # Sacamos las conexiones
            for dicc_NodeDst, dicc_Dist in diccionario_adyacencias.items():

                #   Si no existe, lo creamos
                if not dicc_NodeDst in d_dist:
                    d_dist[dicc_NodeDst] = dicc_Dist + dist     # Nueva Entrada. {v : (dist(prev(v),v) + Dst_Acumulada)}
                    d_prev[dicc_NodeDst] = nodoActual           # Nueva Entrada. {v : prev(v)}
                    Q.put((d_dist[dicc_NodeDst],dicc_NodeDst))  # Metemos elemento en Q. (dist(u,v), v)

                #   Si existe, comprobamos si tiene menor peso que el anterior guardado
                elif d_dist[dicc_NodeDst] > (d_dist[nodoActual] + d_g[nodoActual][dicc_NodeDst]):
                    d_dist[dicc_NodeDst] = d_dist[nodoActual] + d_g[nodoActual][dicc_NodeDst]   # Nueva Entrada. {v : (dist(prev(v),v) + Dst_Acumulada)}
                    d_prev[dicc_NodeDst] = nodoActual           # Nueva Entrada. {v : prev(v)}
                    Q.put((d_dist[dicc_NodeDst],dicc_NodeDst))  # Metemos elemento en Q. (dist(u,v), v)

    return d_dist, d_prev


# In[ ]:


diccDistancias, diccPrevios = dijkstra_d(grafoDicc,0)
print("Diccionario del grafo original:\n", grafoDicc, "\n")
print("Diccionario de Distancias:\n", diccDistancias, "\n")
print("Diccionario de Previos:\n", diccPrevios, "\n")


# In[ ]:


def dijkstra_m(m_g, u):
    """
    Funcion que resuelve el algoritmo de Dijkstra para un nodo de un grafo almacenado
    en un Matriz Numpy de adyacencia.

    Parámetros:
    m_g --> Matriz Numpy de adyacencia de un grafo a aplicar Dijkstra sobre uno de sus nodos
    u ----> Nodo del grafo a aplicar Dijkstra

    Retorno:
    +   Diccionario de distancia de u a K con peso V.
    +   Diccionario de previos donde V es el nodo previo a K.
    """
    
    lstOpenNodes = [] #s    # Lista de nodos abiertos
    d_dist = {} #d          # Diccionario de costes minimos
    d_prev = {} #p          # Diccionario de nodos previos
    lista_adyacencias = []  # Lista de adyacencias
    
    for i in range(len(m_g)):  # Inicializamos la lista entera a False
        lstOpenNodes.append(False)
        
    Q = queue.PriorityQueue()  # Inicializamos la cola de prioridad
    
    d_dist[u] = 0           # K : nodoDst ; V : peso
    Q.put( (d_dist[u],u) )  # [0]: Peso; [1]: NodoDst   # Debe ser asi para que funcione la prioridad
    
    # Mientras la cola de prioridad no esté vacía
    while not Q.empty():
        
        dist_tuple = Q.get()
        dist = dist_tuple[0]
        nodoActual = dist_tuple[1]
        
        # Si el nodo no está cerrado
        if not lstOpenNodes[nodoActual]:
                        
            lstOpenNodes[nodoActual] = True            # Cerramos el nodo
            # lista_adyacencias.append(m_g[nodoActual])  # Sacamos las adyacencias del nodo a analizar
            lista_adyacencias = m_g[nodoActual]
                        
            # Sacamos las conexiones
            contador_nodos=0 # Reseteamos contador
            for nodeDst in lista_adyacencias:
                
                # Si no existe, lo creamos
                if not contador_nodos in d_dist:
                    d_dist[contador_nodos] = lista_adyacencias[contador_nodos] + dist   # Nueva Entrada {v: (dist(prev(v),v) + Dst_Acumulada)}
                    d_prev[contador_nodos] = nodoActual         # Nueva Entrada {v: prev(v)}
                    Q.put( (d_dist[contador_nodos],contador_nodos) )  # Metemos elemento en Q {dist(u,v), v}
                
                # Si existe, comprobamos si tiene menor peso que el anterior guardado
                elif d_dist[contador_nodos] > (d_dist[nodoActual] + m_g[nodoActual][contador_nodos]):
                    d_dist[contador_nodos] = d_dist[nodoActual] + m_g[nodoActual][contador_nodos]  # Nueva Entrada
                    d_prev[contador_nodos] = nodoActual               # Nueva Entrada {v: prev(v)}
                    Q.put( (d_dist[contador_nodos],contador_nodos) )   # Metemos elemento en Q (dist(u,v),v)
                
                # Actualizamos el contador de nodos:
                contador_nodos = contador_nodos + 1
    
    return d_dist, d_prev


# In[ ]:


diccDistancias, diccPrevios = dijkstra_m(grafoMatriz,0)
print("Matriz Numpy del grafo original:\n", grafoMatriz, "\n")
print("Diccionario de Distancias:\n", diccDistancias, "\n")
print("Diccionario de Previos:\n", diccPrevios, "\n")


# In[ ]:


def min_paths(d_prev):
    """
    Función que genera un diccionario con los caminos mínimos de cada uno de los nodos de un determinado grafo.
    Para ello utiliza el diccionario de nodos previos que genera el algoritmo de Dijkstra que hemos desarrollado
    en esta práctica.

    Parámetros:
    d_prev ---------> Diccionario de nodos previos

    Retorno:
    Diccionario con los caminos mínimos de cada uno de los nodos del grafo.
    """
    
    d_path = {}
    lista_previos = []
    
    for node, prev in d_prev.items():
        lista_previos = []
        lista_previos.insert(0,prev)
        d_path[node] = lista_previos
        
    # node2=4, lst=[3]    
    for node2, lst in d_path.items():
        for node3, prev3 in d_prev.items():
            for x in lst:
                if node3 == x:
                    lst.append(prev3)
    
    return d_path


# In[ ]:


def time_dijkstra_m(n_graphs, n_nodes_ini, n_nodes_fin, step, sparse_factor=.25):
    """
    Función que genera un número de grafos con un numero de nodos entre un minimo
    y un máximo conocidos con incrementos de nodos entre ese rango . Despues, para
    cada grafo generado, aplicamos el algoritmo de Dijkstra para cada nodo de cada
    grafo, midiendo el tiempo en el se resuelve el mismo.
    Los grafos están en modo Matriz Numpy.

    Parámetros:
    n_graphs ---------> Número de grados por paso
    n_nodes_ini ------> Número de nodos del paso inicial
    n_nodes_fin ------> Número de nodos del paso final
    step -------------> Incremento de nodos entre los diferentes pasos
    sparse_factor ----> Factor de ramificación de los grafos a generar

    Retorno:
    Lista con los tiempos medios de resolución de Dijkstra para todos los nodos de todos
    los grafos de cada paso.
    """
    
    lista_matrGrafos = [] # Creamos una lista vacia para los grafos em formato matriz
    
    # Creamos n_graphs grafos con los distintos valores entre n_nodes_ini y n_nodes_fin
    # menores estrictamente que n_nodes_fin debido a como funciona range()
    for n_nodes in range(n_nodes_ini, n_nodes_fin, step):
        for j in range(n_graphs):
            newMatrixGraph = rand_matr_pos_graph(n_nodes, sparse_factor)
            lista_matrGrafos.append(newMatrixGraph)
    
    # Creamos los grafos con n_nodes_fin numero de nodos que falta
    for j in range(n_graphs):
        newMatrixGraph = rand_matr_pos_graph(n_nodes_fin, sparse_factor)
        lista_matrGrafos.append(newMatrixGraph)
    
    # Creamos una lista auxiliar vacia para los tiempos de resolución de cada matríz
    listaStep_matrTiempos = []
    # Creamos una lista vacía para los tiempos de resolución de cada matriz
    lista_matrAvgTiempos = []
    # Inicializamos el step actual al mínimo de nodos
    actualNodes_step = n_nodes_ini
    
    for grafoMatr in lista_matrGrafos:
        time_start = time.time()            # Tiempo en segundos al empezar a resolver un grafo
        for vertix in range(grafoMatr.shape[0]):         # Resolvemos el grafo para TODOS los nodos
            dijkstra_m(grafoMatr, vertix)
        time_end = time.time()              # Tiempo en segundos al finalizar de resolver un grafo

        # Calculamos el tiempo medio de resolucion del algoritmo para todos los grafos de un step
        # Por tanto, solo entramos sí el grafo actual es de distinto tamaño al step actual o si
        # estamos en el ultimo grafo, en ese caso calculariamos el tiempo medio de resolución para
        # el ultimo step
        if (len(grafoMatr) != actualNodes_step) or (grafoMatr.all  == lista_matrGrafos[-1].all ):
            
            if grafoMatr.all  == lista_matrGrafos[-1].all :   # Añadimos el ultimo nodo
                listaStep_matrTiempos.append(time_end-time_start)

            avg_tiempo = 0                              # Inicializamos el tiempo medio
            for tiempo in listaStep_matrTiempos:        # Hacemos el sumatorio de todos los tiempos de un step
                avg_tiempo += tiempo
            avg_tiempo /= n_graphs                      # Calculamos la media

            lista_matrAvgTiempos.append(avg_tiempo)     # Añadimos a la lista el nuevo tiempo medio
            actualNodes_step += step                    # Aumentamos el step
            listaStep_matrTiempos = []                  # Limpiamos la lista de tiempos del step

        listaStep_matrTiempos.append(time_end-time_start) # Añadimos el nuevo tiempo a la lista de un step

    return lista_matrAvgTiempos


# In[ ]:


lista_tiemposMatrix = time_dijkstra_m(20, 10, 50, 5, sparse_factor=.8)
print("Lista de Tiempos de Resolucion de Dijkstra para Matrices Numpy (5x5), de 10 a 50 nodos con \5 nodos por intervalo:\n", lista_tiemposMatrix)


# In[ ]:


def time_dijkstra_d(n_graphs, n_nodes_ini, n_nodes_fin, step, sparse_factor=.25):
    """
    Función que genera un número de grafos con un numero de nodos entre un minimo
    y un máximo conocidos con incrementos de nodos entre ese rango. Despues, para
    cada grafo generado, aplicamos el algoritmo de Dijkstra para cada nodo de cada
    grafo, midiendo el tiempo en el se resuelve el mismo.
    Los grafos están en modo Diccionario.

    Parámetros:
    n_graphs -----------> Número de grados por paso
    n_nodes_ini --------> Número de nodos del paso inicial
    n_nodes_fin --------> Número de nodos del paso final
    step ---------------> Incremento de nodos entre los diferentes pasos
    sparse_factor ------> Factor de ramificación de los grafos a generar

    Retorno:
    Lista con los tiempos medios de resolución de Dijkstra para todos los nodos de todos
    los grafos de cada paso.
    """

    lista_diccGrafos = []   # Creamos una lista vacia para los grafos en formato diccionario

    # Creamos n_graphs grafos con los distintos valores entre n_nodes_ini y n_nodes_fin
    # menores estrictamente que n_nodes_fin debido a como funciona range()
    for n_nodes in range(n_nodes_ini, n_nodes_fin, step):
        for j in range(n_graphs):
            newMatrixGraph = rand_matr_pos_graph(n_nodes, sparse_factor)
            newDiccGraph = m_g_2_d_g(newMatrixGraph)
            lista_diccGrafos.append(newDiccGraph)
    # Creamos los grafos con n_nodes_fin numero de nodos que falta
    for j in range(n_graphs):
        newMatrixGraph = rand_matr_pos_graph(n_nodes_fin, sparse_factor)
        newDiccGraph = m_g_2_d_g(newMatrixGraph)
        lista_diccGrafos.append(newDiccGraph)

    listaStep_diccTiempos = []      # Creamos una lista auxiliar vacia para los tiempos de resolución de cada matriz
    lista_diccAvgTiempos = []       # Creamos una lista vacia para los tiempos de resolución de cada matriz
    actualNodes_step = n_nodes_ini  # Inicializamos el step actual al minimo de nodos

    for grafoDicc in lista_diccGrafos:

        time_start = time.time()            # Tiempo en segundos al empezar a resolver un grafo
        for nodo_ini in grafoDicc:          # Resolvemos el grafo para TODOS los nodos
            dijkstra_d(grafoDicc, nodo_ini)
        time_end = time.time()              # Tiempo en segundos al finalizar de resolver un grafo

        # Calculamos el tiempo medio de resolucion del algoritmo para todos los grafos de un step
        # Por tanto, solo entremos sí el grafo actual es de distinto tamaño al step actual o si
        # estamos en el ultimo grafo, en ese caso calculariamos el tiempo medio de resolución para
        # el ultimo step
        if (len(grafoDicc) != actualNodes_step) or (grafoDicc == lista_diccGrafos[-1]):

            if grafoDicc == lista_diccGrafos[-1]:   # Añadimos el ultimo nodo
                listaStep_diccTiempos.append(time_end - time_start)

            avg_tiempo = 0  # Inicializamos el tiempo medio
            for tiempo in listaStep_diccTiempos:    # Hacemos el sumatorio de todos los tiempos de un step
                avg_tiempo += tiempo
            avg_tiempo /= n_graphs                  # Calculamos la media

            lista_diccAvgTiempos.append(avg_tiempo)     # Añadimos a la lista el nuevo tiempo medio
            actualNodes_step += step                    # Aumentamos el step
            listaStep_diccTiempos = []                  # Limpiamos la lista de tiempos del step

        listaStep_diccTiempos.append(time_end - time_start) # Añadimos el nuevo tiempo a la lista de un step

    return lista_diccAvgTiempos


# In[ ]:


lista_tiemposDicc = time_dijkstra_d(20, 10, 50, 5, sparse_factor=.8)
print("Lista de Tiempos de Resolucion de Dijkstra para Diccionarios de listas de \adyacencia (5x5), de 10 a 50 nodos con 5 nodos por intervalo:\n", lista_tiemposDicc)


# In[ ]:


def d_g_2_nx_g(d_g):
    """
    Función que pasa una matriz de adyacencia en formato diccionario de listas de adyacencia a formato de grafo dirigido
    NetworkX.

    Parámetros:
    d_g --> Diccionario de listas de adyacencia a transformar

    Retorno:
    El grafo dirigido NetworkX resultante a partir del dicionario.
    """

    graphNx = nx.DiGraph()

    for nodeOrg, nodesDst in d_g.items():
        for nodeDst, weightEdge in nodesDst.items():
            graphNx.add_edge(nodeOrg, nodeDst, weight = weightEdge)

    return graphNx


# In[ ]:


grafoNx = d_g_2_nx_g(grafoDicc)
print("Diccionario convertida a Grafo NetworkX:\n", grafoNx.adj)


# In[ ]:


def nx_g_2_d_g(nx_g):
    """
    Función que pasa una matriz de adyacencia en formato grafo dirigido NetworkX a formato de dicionario de listas de 
    adyacencia.

    Parámetros:
    nx_g --> Diccionario de listas de adyacencia a transformar

    Retorno:
    El diccionario de listas de adyacencia resultante a partir del grafo dirigido NetworkX.
    """

    d_g = {}

    for nx_VOrg, nx_diccVDsts in nx_g.adj.items():
        d_g[nx_VOrg] = {}
        for nx_VDst, nx_Vatributes in nx_diccVDsts.items():
            d_g[nx_VOrg][nx_VDst] = nx_Vatributes['weight']
            
    return d_g


# In[ ]:


print("Diccionario Original:\n", grafoDicc, "\n")
grafoDicc = nx_g_2_d_g(grafoNx)
print("Grafo NetworkX convertida a Diccionario:\n", grafoDicc)


# In[ ]:


def time_dijkstra_nx(n_graphs, n_nodes_ini, n_nodes_fin, step, sparse_factor=.25):
    """
    Función que genera un número de grafos con un numero de nodos entre un minimo
    y un máximo conocidos con incrementos de nodos entre ese rango. Despues, para
    cada grafo generado, aplicamos el algoritmo de Dijkstra para cada nodo de cada
    grafo, midiendo el tiempo en el se resuelve el mismo.
    Los grafos están en modo NetworkX Graph Data Structure.

    Parámetros:
    n_graphs -----------> Número de grados por paso
    n_nodes_ini --------> Número de nodos del paso inicial
    n_nodes_fin --------> Número de nodos del paso final
    step ---------------> Incremento de nodos entre los diferentes pasos
    sparse_factor ------> Factor de ramificación de los grafos a generar

    Retorno:
    Lista con los tiempos medios de resolución de Dijkstra para todos los nodos de todos
    los grafos de cada paso.
    """

    lista_NxGrafos = [] # Creamos una lista vacia para los grafos en formato NetworkX Graph Data Structure

    # Creamos n_graphs grafos con los distintos valores entre n_nodes_ini y n_nodes_fin
    # menores estrictamente que n_nodes_fin debido a como funciona range()
    for n_nodes in range(n_nodes_ini, n_nodes_fin, step):
        for j in range(n_graphs):
            newMatrixGraph = rand_matr_pos_graph(n_nodes, sparse_factor)
            newDiccGraph = m_g_2_d_g(newMatrixGraph)
            newNxGraph = d_g_2_nx_g(newDiccGraph)
            lista_NxGrafos.append(newNxGraph)
    # Creamos los grafos con n_nodes_fin numero de nodos que falta
    for j in range(n_graphs):
        newMatrixGraph = rand_matr_pos_graph(n_nodes_fin, sparse_factor)
        newDiccGraph = m_g_2_d_g(newMatrixGraph)
        newNxGraph = d_g_2_nx_g(newDiccGraph)
        lista_NxGrafos.append(newNxGraph)

    listaStep_NxTiempos = [] # Creamos una lista auxiliar vacia para los tiempos de resolución de cada grafo
    lista_NxAvgTiempos = [] # Creamos una lista vacia para los tiempos de resolución de cada grafo
    actualNodes_step = n_nodes_ini # Inicializamos el step actual al minimo de nodos

    for grafoNx in lista_NxGrafos:

        time_start = time.time() # Tiempo en segundos al empezar a resolver un grafo
        for nodo_ini in grafoNx: # Resolvemos el grafo para TODOS los nodos
            nx.single_source_dijkstra(grafoNx, nodo_ini)
        time_end = time.time() # Tiempo en segundos al finalizar de resolver un grafo

        # Calculamos el tiempo medio de resolucion del algoritmo para todos los grafos de un step
        # Por tanto, solo entremos sí el grafo actual es de distinto tamaño al step actual o si
        # estamos en el ultimo grafo, en ese caso calculariamos el tiempo medio de resolución para
        # el ultimo step
        if (len(grafoNx) != actualNodes_step) or (grafoNx == lista_NxGrafos[-1]):

            if grafoNx == lista_NxGrafos[-1]: # Añadimos el ultimo nodo
                listaStep_NxTiempos.append(time_end - time_start)

            avg_tiempo = 0 # Inicializamos el tiempo medio
            for tiempo in listaStep_NxTiempos: # Hacemos el sumatorio de todos los tiempos de un step
                avg_tiempo += tiempo
            avg_tiempo /= n_graphs # Calculamos la media

            lista_NxAvgTiempos.append(avg_tiempo) # Añadimos a la lista el nuevo tiempo medio
            actualNodes_step += step # Aumentamos el step
            listaStep_NxTiempos = [] # Limpiamos la lista de tiempos del step

        listaStep_NxTiempos.append(time_end - time_start) # Añadimos el nuevo tiempo a la lista de un step

    return lista_NxAvgTiempos


# In[ ]:


lista_NxAvgT = time_dijkstra_nx(20, 10, 50, 5, sparse_factor=.8)
print("Lista de Tiempos de Resolucion de Dijkstra para Grafos NetworkX de matrices de \adyacencia (5x5), de 10 a 50 nodos con 5 nodos por intervalo:\n", lista_NxAvgT)


# In[ ]:


print("Tiempos por estructura:\n")
print("\t+ Matriz Numpy:\n", lista_tiemposMatrix, "\n")
print("\t+ Diccionario:\n", lista_tiemposDicc, "\n")
print("\t+ Grafo dirigido NetworkX:\n", lista_NxAvgT, "\n")

