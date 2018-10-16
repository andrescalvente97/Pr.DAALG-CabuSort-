def single_source_dijkstra(nx_g, u):
    lstOpenNodes = [] #s    # Lista de nodos abiertos
    d_dist = {} #d          # Diccionario de costes minimos
    d_prev = {} #p          # Diccionario de nodos previos

    for i in range(len(nx_g)):       # Inicializamos la lista entera a False
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
            diccionario_adyacencias = nx_g[nodoActual]   # Sacamos las adyacencias del nodo a analizar

            # Sacamos las conexiones
            for nx_NodeDst, nx_atributes in diccionario_adyacencias.items():

                #   Si no existe, lo creamos
                if not nx_NodeDst in d_dist:
                    d_dist[nx_NodeDst] = nx_atributes['weight'] + dist     # Nueva Entrada. {v : (dist(prev(v),v) + Dst_Acumulada)}
                    d_prev[nx_NodeDst] = nodoActual           # Nueva Entrada. {v : prev(v)}
                    Q.put((d_dist[nx_NodeDst],nx_NodeDst))  # Metemos elemento en Q. (dist(u,v), v)

                #   Si existe, comprobamos si tiene menor peso que el anterior guardado
                elif d_dist[nx_NodeDst] > (d_dist[nodoActual] + nx_g[nodoActual][nx_NodeDst]['weight']):
                    d_dist[nx_NodeDst] = d_dist[nodoActual] + nx_g[nodoActual][nx_NodeDst]['weight']   # Nueva Entrada. {v : (dist(prev(v),v) + Dst_Acumulada)}
                    d_prev[nx_NodeDst] = nodoActual           # Nueva Entrada. {v : prev(v)}
                    Q.put((d_dist[nx_NodeDst],nx_NodeDst))  # Metemos elemento en Q. (dist(u,v), v)

    ##TODO::  ME CALCULA LOS PREVIOS, NO EL CAMINO TOTAL, QUE ES MAS OPTIMO, IR HACIENDO EL DICCIONARIO A MEDIDA QUE
    ##       VAMOS HACIENDO DIJKSTRA O AL FINAL CALCULAR LA RUTA ?¿?¿?¿?¿
    
    return (d_dist, d_prev)

def time_dijkstra_nx(n_graphs, n_nodes_ini, n_nodes_fin, step, sparse_factor=.25):

    lista_NxGrafos = []   # Creamos una lista vacia para los grafos en formato NetworkX Graph Data Structure

    # Creamos n_graphs grafos con los distintos valores entre n_nodes_ini y n_nodes_fin
    # menores estrictamente que n_nodes_fin debido a como funciona range()
    for n_nodes in range(n_nodes_ini, n_nodes_fin, step):
        for j in range(n_graphs):
            newMatrixGraph = rand_matr_pos_graph(n_nodes, sparse_factor)
            newDiccGraph = m_g_2_d_g(newMatrixGraph)
            newNxGraph = d_g_2_nx_g(newNxGraph)
            lista_NxGrafos.append(newNxGraph)
    # Creamos los grafos con n_nodes_fin numero de nodos que falta
    for j in range(n_graphs):
        newMatrixGraph = rand_matr_pos_graph(n_nodes_fin, sparse_factor)
        newDiccGraph = m_g_2_d_g(newMatrixGraph)
        newNxGraph = d_g_2_nx_g(newDiccGraph)
        lista_NxGrafos.append(newNxGraph)

    listaStep_NxTiempos = []        # Creamos una lista auxiliar vacia para los tiempos de resolución de cada grafo
    lista_NxAvgTiempos = []         # Creamos una lista vacia para los tiempos de resolución de cada grafo
    actualNodes_step = n_nodes_ini  # Inicializamos el step actual al minimo de nodos

    for grafoNx in lista_NxGrafos:

        time_start = time.time()            # Tiempo en segundos al empezar a resolver un grafo
        for nodo_ini in grafoNx:            # Resolvemos el grafo para TODOS los nodos
            single_source_dijkstra(grafoNx, nodo_ini)
        time_end = time.time()              # Tiempo en segundos al finalizar de resolver un grafo

        # Calculamos el tiempo medio de resolucion del algoritmo para todos los grafos de un step
        # Por tanto, solo entremos sí el grafo actual es de distinto tamaño al step actual o si
        # estamos en el ultimo grafo, en ese caso calculariamos el tiempo medio de resolución para
        # el ultimo step
        if (len(grafoNx) != actualNodes_step) or (grafoNx == lista_NxGrafos[-1]):

            if grafoNx == lista_nxGrafos[-1]:   # Añadimos el ultimo nodo
                listaStep_NxTiempos.append(time_end - time_start)

            avg_tiempo = 0                          # Inicializamos el tiempo medio
            for tiempo in listaStep_NxTiempos:      # Hacemos el sumatorio de todos los tiempos de un step
                avg_tiempo += tiempo
            avg_tiempo /= n_graphs                  # Calculamos la media

            lista_NxAvgTiempos.append(avg_tiempo)   # Añadimos a la lista el nuevo tiempo medio
            actualNodes_step += step                # Aumentamos el step
            listaStep_NxTiempos = []                # Limpiamos la lista de tiempos del step

        listaStep_NxTiempos.append(time_end - time_start) # Añadimos el nuevo tiempo a la lista de un step

    return lista_NxAvgTiempos
