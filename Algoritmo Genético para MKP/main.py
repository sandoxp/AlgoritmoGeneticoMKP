#Integrantes: Sebastián Sandoval y Joaquín Saldivia

import numpy as np
import random

#Función para leer las instancias
def leer_instancias(archivo):
    with open(archivo, 'r') as file:
        lineas = (linea.strip() for linea in file if linea.strip())
        num_problemas = int(next(lineas))

        instancias = []
        for _ in range(num_problemas):
            n_items, n_dimensiones, indice = map(int, next(lineas).split())

            valores = []
            while len(valores) < n_items:
                valores.extend(map(int, next(lineas).split()))
            valores = np.array(valores[:n_items])

            pesos_dimensiones = []
            for _ in range(n_dimensiones):
                pesos = []
                while len(pesos) < n_items:
                    pesos.extend(map(int, next(lineas).split()))
                pesos_dimensiones.append(np.array(pesos[:n_items]))
            pesos_dimensiones = np.array(pesos_dimensiones)

            capacidades = []
            while len(capacidades) < n_dimensiones:
                capacidades.extend(map(int, next(lineas).split()))
            capacidades = np.array(capacidades[:n_dimensiones])

            instancia = {
                'n_items': n_items,
                'n_dimensiones': n_dimensiones,
                'indice': indice,
                'valores': valores,
                'pesos': pesos_dimensiones,
                'capacidades': capacidades
            }
            instancias.append(instancia)

        return instancias



def generar_individuo_factible(n_items, valores, pesos_dimensiones, capacidades):
    # Calcular la suma de pesos por ítem en todas las dimensiones
    suma_pesos = np.sum(pesos_dimensiones, axis=0)  # Suma de pesos por ítem

    # Calcular la relación valor/peso para cada ítem
    razon_valor_peso = valores / suma_pesos

    # Ordenar los índices de los ítems por relación valor/peso descendente
    indices_ordenados = np.argsort(-razon_valor_peso)

    # Crear un individuo vacío (vector de ceros)
    individuo = np.zeros(n_items, dtype=int)

    # Almacenar el peso acumulado en cada dimensión
    total_pesos_actuales = np.zeros_like(capacidades)

    # Construir el individuo iterando sobre los índices ordenados
    for idx in indices_ordenados:
        item_pesos = pesos_dimensiones[:, idx]

        # Verificar si agregar el ítem mantiene la factibilidad en todas las dimensiones
        if np.all(total_pesos_actuales + item_pesos <= capacidades):
            individuo[idx] = 1
            total_pesos_actuales += item_pesos

    return individuo


def generar_poblacion_inicial(instancia, tamano_poblacion=100):
    n_items = instancia['n_items']
    valores = instancia['valores']
    pesos_dimensiones = instancia['pesos']
    capacidades = instancia['capacidades']

    poblacion = []

    # 75% individuos aleatorios
    for _ in range(tamano_poblacion * 3 // 4 ):
        individuo = np.random.randint(2, size=n_items)
        poblacion.append(individuo)

    # 25% individuos factibles
    for _ in range(tamano_poblacion // 4):
        individuo_factible = generar_individuo_factible(n_items, valores, pesos_dimensiones, capacidades)
        poblacion.append(individuo_factible)

    return np.array(poblacion)



def evaluar_fitness(individuo, valores, pesos_dimensiones, capacidades):
    # Valor total
    valor_total = np.sum(individuo * valores)

    # Pesos acumulados
    pesos_acumulados = np.sum(individuo * pesos_dimensiones, axis=1)

    # Exceso de peso y penalización
    exceso = np.maximum(0, pesos_acumulados - capacidades)
    penalizacion = np.sum(exceso) * np.max(valores / np.sum(pesos_dimensiones, axis=0))

    # Fitness = valor total - penalización
    fitness = valor_total - penalizacion

    return fitness, valor_total, penalizacion, pesos_acumulados




# Función para realizar la selección por torneo
def seleccion_por_torneo(poblacion, fitness_poblacion, tamano_torneo=3):
    # Selecciona aleatoriamente 'tamano_torneo' individuos de la población
    seleccionados = random.sample(range(len(poblacion)), tamano_torneo)

    # Encuentra el índice del mejor individuo dentro del torneo
    mejor = seleccionados[0]
    for i in seleccionados:
        if fitness_poblacion[i] > fitness_poblacion[mejor]:
            mejor = i

    # Retorna el individuo ganador del torneo
    return poblacion[mejor]

# Función para seleccionar múltiples padres
def seleccionar_padres(poblacion, fitness_poblacion, num_padres=2, tamano_torneo=3):
    padres = []
    for _ in range(num_padres):
        padre = seleccion_por_torneo(poblacion, fitness_poblacion, tamano_torneo)
        padres.append(padre)
    return padres

def cruzamiento_dos_puntos(padre1, padre2):
    n_genes = len(padre1)

    # Seleccionar dos puntos de corte aleatorios
    punto1, punto2 = sorted(random.sample(range(n_genes), 2))

    # Crear descendientes combinando los genes de los padres
    hijo1 = np.concatenate((padre1[:punto1], padre2[punto1:punto2], padre1[punto2:]))
    hijo2 = np.concatenate((padre2[:punto1], padre1[punto1:punto2], padre2[punto2:]))

    return hijo1, hijo2


def generar_descendencia(poblacion, fitness_poblacion, tamano_torneo=3, tamano_poblacion=100):

    nueva_poblacion = []

    while len(nueva_poblacion) < tamano_poblacion:
        # Seleccionar dos padres mediante torneo
        padres = seleccionar_padres(poblacion, fitness_poblacion, num_padres=2, tamano_torneo=tamano_torneo)
        padre1, padre2 = padres

        # Cruzar a los padres para generar descendientes
        hijo1, hijo2 = cruzamiento_dos_puntos(padre1, padre2)

        # Añadir los hijos a la nueva población
        nueva_poblacion.append(hijo1)
        nueva_poblacion.append(hijo2)

    # Asegurar que la población no exceda el tamaño deseado
    return np.array(nueva_poblacion[:tamano_poblacion])



def mutar_individuo(individuo, tasa_mutacion=0.002):
    # Crear una máscara de mutación con la misma longitud que el individuo
    mascara_mutacion = np.random.rand(len(individuo)) < tasa_mutacion

    # Aplicar la mutación (cambiar 0 a 1 y viceversa)
    individuo_mutado = np.copy(individuo)
    individuo_mutado[mascara_mutacion] = 1 - individuo_mutado[mascara_mutacion]

    return individuo_mutado

def mutar_poblacion(poblacion, tasa_mutacion=0.002):
    np.set_printoptions(threshold=np.inf)
    return np.array([mutar_individuo(individuo, tasa_mutacion) for individuo in poblacion])


def aplicar_elitismo(poblacion, fitness_poblacion, num_elites=10):
    # Obtener los índices de los mejores individuos según el fitness
    indices_elites = np.argsort(fitness_poblacion)[-num_elites:]

    # Seleccionar los individuos élite
    elites = poblacion[indices_elites]
    return elites


def reparar_individuo(individuo, valores, pesos_dimensiones, capacidades):
    # Calcular los pesos acumulados
    pesos_acumulados = np.sum(individuo * pesos_dimensiones, axis=1)

    # Mientras el individuo sea infactible, seguimos reparándolo
    while np.any(pesos_acumulados > capacidades):
        # Identificar los objetos seleccionados
        objetos_seleccionados = np.where(individuo == 1)[0]

        # Calcular la relación valor/peso para los objetos seleccionados
        razones = valores[objetos_seleccionados] / np.sum(pesos_dimensiones[:, objetos_seleccionados], axis=0)

        # Encontrar el objeto con la peor relación valor/peso
        peor_objeto = objetos_seleccionados[np.argmin(razones)]

        # Eliminar el peor objeto
        individuo[peor_objeto] = 0

        # Recalcular los pesos acumulados
        pesos_acumulados = np.sum(individuo * pesos_dimensiones, axis=1)

    return individuo


def resolver_mkp(instancias, max_generaciones=100, tolerancia_mejora=30, tasa_mutacion=0.02, porcentaje_elitismo=0.1):
    resultados = []  # Para almacenar los resultados de cada instancia
    mejor_valor_global = -float('inf')  # Inicializamos el mejor valor global
    mejor_instancia = None  # Para almacenar el número de la mejor instancia
    mejor_vector_global = None  # Para almacenar el vector de selección de la mejor solución
    mejor_pesos_global = None  # Para almacenar los pesos de la mejor solución
    mejor_capacidades_global = None  # Para almacenar las capacidades de la mejor solución
    valores_seleccionados_global = None  # Para almacenar los valores seleccionados

    for idx, instancia in enumerate(instancias):
        print(f"\nResolviendo instancia {idx + 1}...\n")

        # Generar población inicial
        poblacion = generar_poblacion_inicial(instancia)
        mejor_fitness_historico = -float('inf')
        generaciones_sin_mejora = 0
        num_elites = int(len(poblacion) * porcentaje_elitismo)

        for generacion in range(max_generaciones):
            # Evaluar fitness
            fitness_poblacion = [
                evaluar_fitness(individuo, instancia['valores'], instancia['pesos'], instancia['capacidades'])[0]
                for individuo in poblacion
            ]

            # Encontrar el mejor fitness actual
            mejor_fitness_actual = max(fitness_poblacion)
            mejor_indice = np.argmax(fitness_poblacion)
            mejor_individuo = poblacion[mejor_indice]

            print(f"Generación {generacion + 1}: Mejor fitness = {mejor_fitness_actual:.2f}")

            # Actualizar el mejor fitness histórico
            if mejor_fitness_actual > mejor_fitness_historico:
                mejor_fitness_historico = mejor_fitness_actual
                generaciones_sin_mejora = 0
            else:
                generaciones_sin_mejora += 1

            # Criterio de parada
            if generaciones_sin_mejora >= tolerancia_mejora:
                print(f"Convergencia alcanzada en generación {generacion + 1}.")
                break

            # Elitismo: Conservar los mejores individuos
            elites = aplicar_elitismo(poblacion, fitness_poblacion, num_elites)

            # Generar descendencia
            nueva_poblacion = generar_descendencia(
                poblacion,
                fitness_poblacion,
                tamano_torneo=3,
                tamano_poblacion=len(poblacion) - num_elites
            )

            # Aplicar mutación
            nueva_poblacion_mutada = mutar_poblacion(nueva_poblacion, tasa_mutacion)

            # Actualizar la población combinando élites y descendientes
            poblacion = np.vstack((elites, nueva_poblacion_mutada))

        # Evaluar la mejor solución final
        fitness, valor_total, penalizacion, pesos_acumulados = evaluar_fitness(
            mejor_individuo, instancia['valores'], instancia['pesos'], instancia['capacidades']
        )

        # Mostrar la mejor solución antes de la reparación si es infactible
        if not np.all(pesos_acumulados <= instancia['capacidades']):
            print("La mejor solución no es factible. Reparando...")
            print(f"Vector de selección antes de reparación: {mejor_individuo}")
            print(f"Valor total antes de reparación: {valor_total}")
            print(f"Penalización antes de reparación: {penalizacion:.2f}")
            print("Pesos obtenidos / Capacidades por dimensión (antes de reparación):")
            for dim in range(len(pesos_acumulados)):
                print(f"  Dimensión {dim + 1}: {pesos_acumulados[dim]} / {instancia['capacidades'][dim]}")

            # Reparar el individuo
            mejor_individuo = reparar_individuo(mejor_individuo, instancia['valores'], instancia['pesos'], instancia['capacidades'])
            fitness, valor_total, penalizacion, pesos_acumulados = evaluar_fitness(
                mejor_individuo, instancia['valores'], instancia['pesos'], instancia['capacidades']
            )

        # Mostrar detalles finales después de reparación
        print(f"Vector de selección después de reparación: {mejor_individuo}")
        print(f"Valor total después de reparación: {valor_total}")
        print(f"Fitness final: {fitness:.2f}")
        print("Pesos obtenidos / Capacidades por dimensión (después de reparación):")
        for dim in range(len(pesos_acumulados)):
            print(f"  Dimensión {dim + 1}: {pesos_acumulados[dim]} / {instancia['capacidades'][dim]}")

        # Guardar resultados de la instancia
        resultados.append({
            "instancia": idx + 1,
            "vector_seleccion": np.copy(mejor_individuo), 
            "fitness": fitness,
            "valor_total": valor_total,
            "penalizacion": penalizacion,
            "pesos_acumulados": np.copy(pesos_acumulados), 
            "capacidades": instancia['capacidades']
        })

        # Actualizar el mejor valor global si corresponde
        if valor_total > mejor_valor_global:
            mejor_valor_global = valor_total
            mejor_instancia = idx + 1
            mejor_vector_global = np.copy(mejor_individuo)  
            mejor_pesos_global = np.copy(pesos_acumulados) 
            mejor_capacidades_global = instancia['capacidades']
            valores_seleccionados_global = instancia['valores'][mejor_individuo == 1]

    # Mostrar resumen final
    print("\nResolución finalizada.")
    print(f"Mejor instancia: N° {mejor_instancia}")
    print(f"Mejor valor total después de reparación de todas las instancias: {mejor_valor_global}")
    print(f"Vector de selección después de reparación: {np.array2string(mejor_vector_global, separator=',')}")
    print("Pesos obtenidos / Capacidades por dimensión (después de reparación):")
    for dim in range(len(mejor_pesos_global)):
        print(f"  Dimensión {dim + 1}: {mejor_pesos_global[dim]} / {mejor_capacidades_global[dim]}")
    print(f"Valores de variables seleccionadas: {valores_seleccionados_global}")

    return resultados




# Main
if __name__ == "__main__":
    archivo_instancias = 'instancia1.txt'  # Ruta del archivo con las instancias
    instancias = leer_instancias(archivo_instancias)  # Leer las instancias
    resolver_mkp(instancias)  # Resolver el problema MKP