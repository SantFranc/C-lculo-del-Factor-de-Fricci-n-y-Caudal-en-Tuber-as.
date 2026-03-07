# -*- coding: utf-8 -*-
"""
Este módulo implementa un algoritmo iterativo para calcular el factor de fricción (f)
y el caudal volumétrico (Q) de un fluido en una tubería, basándose en el balance
de energía (Ecuación de Darcy-Weisbach).

El problema se resuelve mediante Sustitución Sucesiva, iterando entre el caudal (Q)
y el factor de fricción (f), ya que ambas variables son interdependientes.

Created on Tue Nov 18 18:09:58 2025

@author: Santiago
"""

import numpy # Importa la biblioteca NumPy para operaciones matemáticas eficientes

# ------------------------------------------------------------------------------
# FUNCIONES DE CÁLCULO DE NÚMEROS ADIMENSIONALES
# ------------------------------------------------------------------------------

def numero_reynolds(densidad: float, diametro: float, velocidad: float, viscosidad: float) -> float:
    """Calcula el Número de Reynolds (Re) que determina el régimen de flujo."""
    
    # Aplica la fórmula Re = (rho * d * v) / mu
    n_reynolds = (densidad * diametro * velocidad) / viscosidad
    
    return n_reynolds

def numero_reynolds_limite(rugosidad: float, diametro: float) -> float:
    """
    Calcula el Número de Reynolds Límite (Re_lim) que separa el flujo turbulento
    liso del flujo turbulento rugoso.
    """
    
    # Si la rugosidad es cero (tubería lisa), el límite es infinito (siempre se considera liso).
    if rugosidad == 0:
        return float('inf')
    
    # Constantes de la expresión que calcula Re_lim basado en la rugosidad relativa
    COEFICIENTE_EXPRESION = 484.29008007
    EXPONENTE_VARIABLE_ABS = -1.131550475
    
    # Aplica la fórmula de Re_lim basada en la rugosidad relativa (epsilon/diametro)
    n_reynolds_lim = COEFICIENTE_EXPRESION * (abs(rugosidad / diametro))**EXPONENTE_VARIABLE_ABS
    
    return n_reynolds_lim

# ------------------------------------------------------------------------------
# CONSTANTES DE CONVERGENCIA Y LÍMITES
# ------------------------------------------------------------------------------

TOLERANCIA = 1e-6 # Define la precisión requerida para la convergencia de las iteraciones.
MAXIMAS_ITERACIONES = 50 # Límite máximo de iteraciones para evitar ciclos infinitos.

# ------------------------------------------------------------------------------
# FUNCIONES DE CÁLCULO DEL FACTOR DE FRICCIÓN (f)
# ------------------------------------------------------------------------------

def ff_flujo_turbulento_liso(n_reynolds: float) -> float:
    """
    Calcula el factor de fricción para flujo turbulento en tuberías lisas (rugosidad = 0).
    Resuelve implícitamente la ecuación de Kármán-Nikuradse mediante Sustitución Sucesiva.
    """
    
    # Valor inicial (aproximación de Blasius) para iniciar la iteración.
    factor_friccion_anterior = 0.3164 / (n_reynolds**0.25)
    
    iteraciones = 0
    
    # Ciclo de iteración local de Sustitución Sucesiva
    while iteraciones < MAXIMAS_ITERACIONES:

        # Aplica la ecuación implícita de Kármán-Nikuradse para 1/sqrt(f)
        expresion = -0.8 + 2 * numpy.log10(numpy.abs(n_reynolds * numpy.sqrt(factor_friccion_anterior)))
        # Se calcula f_nuevo a partir de 1/sqrt(f)
        factor_friccion_nuevo = expresion**(-2)
        
        # Comprueba la convergencia por diferencia absoluta.
        if numpy.abs(factor_friccion_nuevo - factor_friccion_anterior) < TOLERANCIA:
    
            return factor_friccion_nuevo # Retorna el valor si converge.
    
        factor_friccion_anterior = factor_friccion_nuevo # Actualiza para la siguiente iteración.
        iteraciones += 1

    return factor_friccion_anterior # Retorna el último valor si no converge.

def ff_flujo_turbulento_rugoso(n_reynolds: float, diametro: float, rugosidad: float) -> float:
    """
    Calcula el factor de fricción para flujo turbulento en la zona de transición rugosa.
    Resuelve implícitamente la ecuación de Colebrook-White mediante Sustitución Sucesiva.
    """
    
    # Valor inicial basado en la correlación de Swamee-Jain o similar (aproximación explícita)
    termino_numerico_rugosidad = rugosidad / (3.7 * diametro)
    termino_numerico_reynolds = 5.74 / (n_reynolds**0.9)
    factor_friccion_anterior = 0.25 / (numpy.log10(termino_numerico_rugosidad + termino_numerico_reynolds)**2)
    
    iteraciones = 0
    
    # Ciclo de iteración local de Sustitución Sucesiva (Colebrook-White)
    while iteraciones < MAXIMAS_ITERACIONES:
        
        # Componentes del argumento del logaritmo de Colebrook-White
        termino_rugosidad = rugosidad / (3.71 * diametro)
        termino_reynolds = 2.51 / (n_reynolds * numpy.sqrt(factor_friccion_anterior))
        
        argumento_logaritmo = numpy.abs(termino_rugosidad + termino_reynolds)
        # Aplica la expresión de Colebrook-White para 1/sqrt(f)
        expresion = -2 * numpy.log10(argumento_logaritmo)
        
        factor_friccion_nuevo = expresion**(-2) # Calcula f_nuevo
        
        # Comprueba la convergencia por diferencia absoluta.
        if numpy.abs(factor_friccion_nuevo - factor_friccion_anterior) < TOLERANCIA:
                
            return factor_friccion_nuevo # Retorna el valor si converge.
            
        factor_friccion_anterior = factor_friccion_nuevo # Actualiza para la siguiente iteración.
        iteraciones += 1
        
    return factor_friccion_anterior # Retorna el último valor si no converge.

def factor_friccion(densidad: float, diametro: float, velocidad: float, viscosidad: float, rugosidad: float) -> float:
    """
    Función principal de decisión que selecciona la ecuación de f en base al régimen de flujo (Re).
    """
    
    # Calcula los números de Reynolds necesarios
    n_reynolds = numero_reynolds(densidad, diametro, velocidad, viscosidad)
    n_reynolds_lim = numero_reynolds_limite(rugosidad, diametro)
    
    # Bloque de Decisión 1: Flujo Laminar (Re < 2400)
    if n_reynolds < 2400:
        
        f_friccion = 64 / n_reynolds # Ecuación de Poiseuille (explícita)
        
    # Bloque de Decisión 2: Flujo Turbulento (Re > 4200)
    elif n_reynolds > 4200:
        
        # Decisión A: Tubería lisa (rugosidad == 0)
        if rugosidad == 0:
            
            f_friccion = ff_flujo_turbulento_liso(n_reynolds) # Llama a la iteración de Kármán-Nikuradse
            
        # Decisión B: Tubería rugosa (rugosidad != 0)
        else:
            
            # Decisión B.1: Zona Completamente Rugosa (Re >= Re_lim)
            if n_reynolds >= n_reynolds_lim:
                
                # Ecuación de Nikuradse (explícita)
                expresion = 2 * numpy.log10(abs(diametro / rugosidad)) + 1.14
                f_friccion = expresion**(-2)
                
            # Decisión B.2: Zona de Transición Rugosa (Re < Re_lim)
            elif n_reynolds < n_reynolds_lim:
                
                f_friccion = ff_flujo_turbulento_rugoso(n_reynolds, diametro, rugosidad) # Llama a la iteración de Colebrook-White
                
    # Bloque de Decisión 3: Flujo de Transición (2400 < Re < 4200)
    else:
        
        # Genera una excepción para indicar que el flujo es indeterminado en esta zona.
        raise ValueError('Imposible de determinar: flujo en transición.')
        
    return f_friccion

# ------------------------------------------------------------------------------
# FUNCIONES DE CÁLCULO DE CAUDAL Y GEOMETRÍA
# ------------------------------------------------------------------------------

# Valor de la aceleración gravitacional local para el Distrito de Medellín (según enunciado)
ACELERACION_GRAVITACIONAL_LOCAL = 9.76

def coeficiente_conduccion_geometrica(longitud: float, diametro: float) -> float:
    """Calcula el coeficiente de conducción geométrica 'c'."""
    
    # Aplica la fórmula c = 8 * L / (pi^2 * g * d^5)
    cc_geometrica = (8 * longitud) / ((numpy.pi)**2 * ACELERACION_GRAVITACIONAL_LOCAL * diametro**5)
    
    return cc_geometrica

def caudal_f_friccion(diferencia_altura: float, f_friccion: float, cc_geometrica: float) -> float:
    """
    Calcula el caudal (Q) usando la Ecuación de Darcy-Weisbach simplificada (Ec. 7 del informe).
    """
    
    # Protección: Evita la división por cero o la raíz cuadrada de números negativos.
    if (f_friccion * cc_geometrica) <= 0 or diferencia_altura <= 0:
        
        return 0
    
    # Aplica la fórmula Q = sqrt(Delta_H / (f * c))
    caudal = numpy.sqrt(diferencia_altura / (f_friccion * cc_geometrica))
    
    return caudal

# ------------------------------------------------------------------------------
# FUNCIÓN DE ENTRADA DE DATOS DEL USUARIO
# ------------------------------------------------------------------------------

def obtener_parametros():
    """Solicita y valida los parámetros necesarios al usuario."""
    
    # Diccionario que define las variables y los mensajes de solicitud
    parametros_necesarios = {
        'densidad': ('Densidad [kg/m^3]', float),
        'viscosidad': ('Viscosidad [Pa*s]', float),
        'diametro': ('Diámetro [m]', float),
        'rugosidad': ('Rugosidad [m]', float),
        'longitud': ('Longitud [m]', float),
        'diferencia_altura': ('Diferencia de altura [m]', float),
        'v_inicial': ('Velocidad Inicial [m/s]', float),
    }
    
    datos_usuario = {}
    
    # Itera sobre cada parámetro para solicitar la entrada
    for nombre_var, (mensaje, tipo_dato) in parametros_necesarios.items():
        while True:
            try:
                
                # Solicita la entrada al usuario
                entrada = input(f'{mensaje}: ')
                # Intenta convertir la entrada al tipo de dato especificado (float)
                valor = tipo_dato(entrada)
                
                # Validación básica: asegura que la mayoría de los valores no sean negativos.
                if valor < 0 and nombre_var not in ('rugosidad',):
                    print('Entrada inválida.')
                    continue # Vuelve a pedir el valor.
                    
                datos_usuario[nombre_var] = valor # Almacena el valor validado.
                break # Sale del bucle interno si la entrada es válida.
                
            except ValueError:
                # Captura el error si la conversión a float falla (ej. si ingresa texto)
                print(f'Entrada inválida. Debe ser un número tipo "{tipo_dato.__name__}".')
                
    return datos_usuario

# ------------------------------------------------------------------------------
# FUNCIÓN PRINCIPAL DE RESOLUCIÓN ITERATIVA
# ------------------------------------------------------------------------------

def main():
    """
    Función principal que implementa el ciclo de Sustitución Sucesiva global para
    resolver la dependencia circular entre caudal y factor de fricción.
    """
    
    # Obtiene todos los datos del usuario
    parametros = obtener_parametros()
    
    # Asigna las variables desde el diccionario de parámetros
    densidad = parametros['densidad']
    viscosidad = parametros['viscosidad']
    diametro = parametros['diametro']
    rugosidad = parametros['rugosidad']
    longitud = parametros['longitud']
    diferencia_altura = parametros['diferencia_altura']
    velocidad_inicial = parametros['v_inicial']
    
    # Cálculos iniciales de propiedades geométricas
    area = (numpy.pi * diametro**2) / 4
    cc_geometrica = coeficiente_conduccion_geometrica(longitud, diametro)
    
    # Inicialización del ciclo iterativo global
    velocidad_actual = velocidad_inicial # Se usa para calcular el primer Re
    f_friccion_anterior = 1.0 # Valor inicial de f_friccion
    
    caudal_anterior = velocidad_actual * area # Caudal inicial para el criterio de convergencia
    
    # Ciclo global de Sustitución Sucesiva
    for i in range(MAXIMAS_ITERACIONES):
        
        # 1. Recalcula el factor de fricción (f)
        f_friccion_nuevo = factor_friccion(densidad, diametro, velocidad_actual, viscosidad, rugosidad)
        # 2. Recalcula el caudal (Q) usando el nuevo f
        caudal_nuevo = caudal_f_friccion(diferencia_altura, f_friccion_nuevo, cc_geometrica)
        
        # --- Criterio de Convergencia Relativa para f ---
        if f_friccion_nuevo != 0:
            
            # Comprueba si el cambio relativo en f está dentro de la tolerancia
            convergencia_ff = numpy.abs(f_friccion_nuevo - f_friccion_anterior) / numpy.abs(f_friccion_nuevo) < TOLERANCIA
            
        else:
            
            convergencia_ff = False
            
        # --- Criterio de Convergencia Relativa para Q ---
        if caudal_nuevo != 0:
            
            # Comprueba si el cambio relativo en Q está dentro de la tolerancia
            convergencia_caudal = numpy.abs(caudal_nuevo - caudal_anterior) / numpy.abs(caudal_nuevo) < TOLERANCIA
            
        else:
            
            convergencia_caudal = False
            
        # La convergencia se alcanza si converge f O converge Q
        if convergencia_ff or convergencia_caudal:
            
            # Asigna los valores finales y rompe el ciclo
            f_friccion_final = f_friccion_nuevo
            caudal_final_iterado = caudal_nuevo
            break
        
        # Actualiza los valores anteriores para la siguiente iteración
        f_friccion_anterior = f_friccion_nuevo
        caudal_anterior = caudal_nuevo
        
        # 3. Recalcula la velocidad (v) para el siguiente cálculo de Re
        velocidad_actual = caudal_nuevo / area
        
    # Bloque 'else' del for: se ejecuta si el ciclo termina sin un 'break'
    else:
        
        # Si se alcanzó el límite de iteraciones, usa los últimos valores calculados
        f_friccion_final = f_friccion_nuevo
        caudal_final_iterado = caudal_nuevo
        
    # Cálculos finales fuera del ciclo de iteración
    caudal_final = caudal_final_iterado
    velocidad_final = caudal_final / area
    n_reynolds_final = numero_reynolds(densidad, diametro, velocidad_final, viscosidad)
    
    # Muestra los resultados finales al usuario
    print(f'''
---------------------------------
- Número de Reynolds: {n_reynolds_final:.3f}
- Factor de Fricción: {f_friccion_final:.6f}
- Velocidad: {velocidad_final:.6f} m/s
- Caudal: {caudal_final:.6f} m^3/s
---------------------------------''')

# ------------------------------------------------------------------------------
# PUNTO DE ENTRADA DEL PROGRAMA
# ------------------------------------------------------------------------------

if __name__ == '__main__':

    main() # Llama a la función principal para iniciar la ejecución del programa