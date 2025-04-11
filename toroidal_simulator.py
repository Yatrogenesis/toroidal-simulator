#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulador de Sistemas Toroidales Coherentes con Acoplamiento Métrico-Electromagnético
Basado en el artículo: "Falsifiable Metric-Electromagnetic Coupling in Coherent Toroidal Systems"
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.special import jv, yv  # Funciones de Bessel
from scipy.special import lpmv  # Funciones de Legendre asociadas
from scipy.constants import epsilon_0, mu_0, c, hbar, G, e, m_e, alpha
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import pandas as pd
from dataclasses import dataclass, field
import json
import time
import os
from typing import Dict, List, Tuple, Optional, Union, Callable
import warnings
import logging

# Configuración de registro
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ToroidalSimulator")

# Suprimir advertencias específicas
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in sqrt")

class Constants:
    """Constantes físicas fundamentales y derivadas para la simulación."""
    
    # Constantes fundamentales (SI)
    EPSILON_0 = epsilon_0  # Permitividad del vacío
    MU_0 = mu_0            # Permeabilidad del vacío
    SPEED_OF_LIGHT = c     # Velocidad de la luz
    PLANCK_CONSTANT = hbar # Constante de Planck reducida
    GRAV_CONSTANT = G      # Constante gravitacional
    ELEMENTARY_CHARGE = e  # Carga elemental
    ELECTRON_MASS = m_e    # Masa del electrón
    FINE_STRUCTURE = alpha # Constante de estructura fina
    
    # Constantes derivadas
    SCHWINGER_LIMIT = m_e**2 * c**3 / (e * hbar)  # Campo eléctrico crítico
    PLANCK_ENERGY_DENSITY = c**7 / (hbar * G**2)  # Densidad de energía de Planck
    COMPTON_WAVELENGTH = hbar / (m_e * c)         # Longitud de onda Compton del electrón
    
    # Constantes para el sistema toroidal
    @staticmethod
    def default_coupling_constants():
        """Devuelve los valores predeterminados para las constantes de acoplamiento."""
        # Valores basados en la sección 3.2 del artículo
        return {
            'xi_1': 1.5e-24,  # Acoplamiento helicidad fluido-campo (m³/kg)
            'xi_2': 1.4e-16,  # Acoplamiento de divergencia (m³/kg)
            'xi_3': 2.8e-45,  # Acoplamiento métrico-campo (m³/J)
            'xi_4': 1.1e-16,  # Acoplamiento helicidad acústico-electromagnético (m³/J)
            'xi_5': 5.3e-19,  # Acoplamiento helicidad advectivo (m⁴/kg·s)
            'kappa_T': 4.22e-32  # Constante de acoplamiento regularizada (m⁵/J)
        }


@dataclass
class MaterialProperties:
    """Propiedades de materiales para la simulación."""
    
    name: str
    type: str  # 'superconductor', 'conductive_fluid', 'metamaterial'
    
    # Propiedades básicas
    density: float = 0.0  # kg/m³
    conductivity: float = 0.0  # S/m
    
    # Propiedades específicas de superconductores
    critical_temperature: float = 0.0  # K
    critical_field: float = 0.0  # T
    surface_resistance: float = 0.0  # Ohms a una frecuencia específica
    london_penetration_depth: float = 0.0  # m
    
    # Propiedades de fluidos
    viscosity: float = 0.0  # Pa·s
    melting_point: float = 0.0  # °C
    speed_of_sound: float = 0.0  # m/s
    
    # Propiedades electromagnéticas
    permittivity_r: float = 1.0  # Permitividad relativa
    permeability_r: float = 1.0  # Permeabilidad relativa
    
    # Propiedades metamateriales
    enhancement_factor: float = 1.0  # Factor de mejora para metamateriales
    resonant_frequency: float = 0.0  # Hz
    
    def __str__(self):
        return f"{self.name} ({self.type})"
    
    @classmethod
    def load_material_library(cls):
        """Carga una biblioteca de materiales predefinidos basados en el artículo."""
        library = {}
        
        # Superconductores (Tabla 2 del artículo)
        library["Nb"] = cls(
            name="Niobio",
            type="superconductor",
            critical_temperature=9.3,  # K
            critical_field=0.82,  # T
            surface_resistance=150,  # Ohms a 10 GHz, 20K
            density=8570,  # kg/m³
            london_penetration_depth=39e-9  # m
        )
        
        library["Nb3Sn"] = cls(
            name="Niobio-Estaño",
            type="superconductor",
            critical_temperature=18.3,  # K
            critical_field=24.5,  # T
            surface_resistance=35,  # Ohms a 10 GHz, 20K
            density=8900,  # kg/m³
            london_penetration_depth=85e-9  # m
        )
        
        library["YBCO"] = cls(
            name="YBCO",
            type="superconductor",
            critical_temperature=93,  # K
            critical_field=120,  # T
            surface_resistance=10,  # Ohms a 10 GHz, 20K
            density=6380,  # kg/m³
            london_penetration_depth=150e-9  # m
        )
        
        library["MgB2"] = cls(
            name="Diboruro de Magnesio",
            type="superconductor",
            critical_temperature=39,  # K
            critical_field=35,  # T
            surface_resistance=25,  # Ohms a 10 GHz, 20K
            density=2570,  # kg/m³
            london_penetration_depth=180e-9  # m
        )
        
        # Fluidos conductivos (Tabla 3 del artículo)
        library["Mercury"] = cls(
            name="Mercurio",
            type="conductive_fluid",
            conductivity=1.0e6,  # S/m
            density=13534,  # kg/m³
            melting_point=-38.8,  # °C
            viscosity=1.53e-3,  # Pa·s
            speed_of_sound=1450  # m/s
        )
        
        library["Galinstan"] = cls(
            name="Galinstan",
            type="conductive_fluid",
            conductivity=3.46e6,  # S/m
            density=6440,  # kg/m³
            melting_point=-19.0,  # °C
            viscosity=2.4e-3,  # Pa·s
            speed_of_sound=2950  # m/s
        )
        
        library["EGaIn"] = cls(
            name="EGaIn",
            type="conductive_fluid",
            conductivity=3.4e6,  # S/m
            density=6250,  # kg/m³
            melting_point=15.5,  # °C
            viscosity=1.99e-3,  # Pa·s
            speed_of_sound=2740  # m/s
        )
        
        library["NaK"] = cls(
            name="Sodio-Potasio",
            type="conductive_fluid",
            conductivity=2.88e6,  # S/m
            density=866,  # kg/m³
            melting_point=-12.6,  # °C
            viscosity=0.52e-3,  # Pa·s
            speed_of_sound=2300  # m/s
        )
        
        # Metamateriales (Sección 6.3 del artículo)
        library["SRR_CELC"] = cls(
            name="SRR-CELC Metamaterial",
            type="metamaterial",
            enhancement_factor=7.3,  # De la ecuación 69
            resonant_frequency=5.64e9,  # Hz, inferido del artículo
            permittivity_r=-3.5,  # Valor negativo característico
            permeability_r=-1.8,  # Valor negativo característico
            density=8900  # kg/m³, estimado basado en materiales comunes
        )
        
        return library


class GeometryGenerator:
    """Genera geometrías toroidales y pseudotoroidales según lo descrito en la sección 2.1."""
    
    def __init__(self, R=1.0, a=0.3, r0=0.2, z0=0.0, alpha=1.0, beta=10.0, epsilon=0.05, Rt=0.7):
        """
        Inicializa el generador de geometría con los parámetros especificados.
        
        Args:
            R: Radio mayor del toroide
            a: Radio menor del toroide
            r0: Radio de referencia para la cámara parabólica
            z0: Posición z de referencia para la cámara parabólica
            alpha: Coeficiente parabólico
            beta: Controla la pendiente de la transición
            epsilon: Parámetro de regularización para la transición
            Rt: Radio de transición entre geometrías
        """
        self.R = R
        self.a = a
        self.r0 = r0
        self.z0 = z0
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.Rt = Rt
        logger.info(f"Geometría inicializada con R={R}, a={a}")
    
    def standard_torus(self, theta, phi):
        """
        Genera coordenadas para un toroide estándar usando la ecuación (1) del artículo.
        
        Args:
            theta, phi: Coordenadas angulares paramétricas del toroide
            
        Returns:
            Coordenadas cartesianas (x, y, z)
        """
        x = (self.R + self.a * np.cos(theta)) * np.cos(phi)
        y = (self.R + self.a * np.cos(theta)) * np.sin(phi)
        z = self.a * np.sin(theta)
        return x, y, z
    
    def parabolic_chamber(self, r, phi):
        """
        Genera coordenadas para la cámara parabólica usando la ecuación (2) del artículo.
        
        Args:
            r: Coordenada radial
            phi: Coordenada angular
            
        Returns:
            Coordenadas cartesianas (x, y, z)
        """
        z = self.alpha * (r - self.r0)**2 + self.z0
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        return x, y, z
    
    def blending_function(self, r):
        """
        Función de mezcla para la transición suave entre geometrías, ecuación (3).
        
        Args:
            r: Coordenada radial (distancia desde el origen)
            
        Returns:
            Valor de la función de mezcla entre 0 y 1
        """
        return 0.5 * (1 + np.tanh(self.beta * (r - self.Rt) / self.epsilon))
    
    def pseudotoroidal_geometry(self, theta_vals, phi_vals):
        """
        Genera la geometría pseudotoroidal completa combinando toroide y cámara parabólica.
        
        Args:
            theta_vals: Array de valores para coordenada theta
            phi_vals: Array de valores para coordenada phi
            
        Returns:
            Grids de coordenadas X, Y, Z para la geometría completa
        """
        theta_grid, phi_grid = np.meshgrid(theta_vals, phi_vals)
        
        # Generar coordenadas toroidales estándar
        x_torus, y_torus, z_torus = self.standard_torus(theta_grid, phi_grid)
        
        # Calcular radios para cada punto (usado en la función de mezcla)
        r_grid = np.sqrt(x_torus**2 + y_torus**2)
        
        # Para la cámara parabólica, usamos r como una fracción del radio mayor
        r_parab_vals = np.linspace(0, self.r0, len(theta_vals))
        r_parab_grid, phi_parab_grid = np.meshgrid(r_parab_vals, phi_vals)
        
        x_parab, y_parab, z_parab = self.parabolic_chamber(r_parab_grid, phi_parab_grid)
        
        # Calcular la función de mezcla para cada punto
        blend = self.blending_function(r_grid)
        
        # Interpolar los valores parabólicos al tamaño de la malla toroidal
        from scipy.interpolate import griddata
        points = np.column_stack((r_parab_grid.ravel(), phi_parab_grid.ravel()))
        x_parab_interp = griddata(
            points, x_parab.ravel(), (r_grid.ravel(), phi_grid.ravel()), method='linear'
        ).reshape(r_grid.shape)
        y_parab_interp = griddata(
            points, y_parab.ravel(), (r_grid.ravel(), phi_grid.ravel()), method='linear'
        ).reshape(r_grid.shape)
        z_parab_interp = griddata(
            points, z_parab.ravel(), (r_grid.ravel(), phi_grid.ravel()), method='linear'
        ).reshape(r_grid.shape)
        
        # Aplicar la función de mezcla según la ecuación (4)
        X = (1 - blend) * x_parab_interp + blend * x_torus
        Y = (1 - blend) * y_parab_interp + blend * y_torus
        Z = (1 - blend) * z_parab_interp + blend * z_torus
        
        return X, Y, Z
    
    def visualize_geometry(self, resolution=50):
        """
        Visualiza la geometría pseudotoroidal completa.
        
        Args:
            resolution: Número de puntos en cada dirección para la visualización
            
        Returns:
            Figure y axes de matplotlib con la visualización 3D
        """
        theta_vals = np.linspace(0, 2*np.pi, resolution)
        phi_vals = np.linspace(0, 2*np.pi, resolution)
        
        X, Y, Z = self.pseudotoroidal_geometry(theta_vals, phi_vals)
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Graficar la superficie con un colormap que resalte la estructura
        surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, 
                              linewidth=0, antialiased=True, alpha=0.8)
        
        # Añadir componentes clave según la Figura 1 del artículo
        # Marcar los puertos de excitación RF (4 puertos equidistantes)
        port_phi = np.linspace(0, 2*np.pi, 5)[:-1]  # 4 ángulos equidistantes
        port_theta = np.pi/4  # Posición angular en la sección transversal
        
        port_x, port_y, port_z = self.standard_torus(port_theta, port_phi)
        ax.scatter(port_x, port_y, port_z, color='red', s=100, 
                  label='Puertos de excitación RF')
        
        # Marcar ubicaciones de sensores
        sensor_phi = np.linspace(0, 2*np.pi, 9)[:-1]  # 8 ángulos equidistantes
        sensor_theta = 3*np.pi/4  # Posición angular diferente para sensores
        
        sensor_x, sensor_y, sensor_z = self.standard_torus(sensor_theta, sensor_phi)
        ax.scatter(sensor_x, sensor_y, sensor_z, color='green', s=80, 
                  label='Ubicaciones de sensores')
        
        # Añadir leyenda y etiquetas
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Geometría Pseudotoroidal con Cámara de Resonancia Parabólica')
        ax.legend()
        
        # Añadir barra de colores para mostrar la transición
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        cbar.set_label('Intensidad de campo (relativa)')
        
        return fig, ax
    
    def generate_cross_section(self, phi_value=0, resolution=100):
        """
        Genera una vista de sección transversal de la geometría a un ángulo phi específico.
        
        Args:
            phi_value: Valor de phi para la sección transversal
            resolution: Resolución de la malla
            
        Returns:
            Arrays r (coordenada radial) y z (altura)
        """
        theta_vals = np.linspace(0, 2*np.pi, resolution)
        phi_vals = np.array([phi_value])
        
        X, Y, Z = self.pseudotoroidal_geometry(theta_vals, phi_vals)
        
        # Extraer el corte a phi_value
        X_slice = X[:, 0]
        Y_slice = Y[:, 0]
        Z_slice = Z[:, 0]
        
        # Convertir a coordenadas radiales para la sección transversal
        r_slice = np.sqrt(X_slice**2 + Y_slice**2)
        
        return r_slice, Z_slice
    
    def visualize_cross_section(self, phi_value=0, resolution=100):
        """
        Visualiza una sección transversal de la geometría pseudotoroidal.
        
        Args:
            phi_value: Valor de phi para la sección transversal
            resolution: Resolución de la malla
            
        Returns:
            Figure y axes de matplotlib con la visualización 2D
        """
        r_slice, Z_slice = self.generate_cross_section(phi_value, resolution)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(r_slice, Z_slice, 'b-', linewidth=2)
        ax.fill(r_slice, Z_slice, alpha=0.2, color='blue')
        
        # Marcar componentes clave
        ax.annotate('Cubierta superconductora', xy=(self.R + self.a/2, self.a/2), 
                  xytext=(self.R + self.a*1.5, self.a*1.5), 
                  arrowprops=dict(facecolor='black', shrink=0.05))
        
        ax.annotate('Cámara de resonancia\nparabólica', xy=(self.r0/2, self.z0), 
                  xytext=(self.r0/2, self.z0 - 0.5*self.a), 
                  arrowprops=dict(facecolor='black', shrink=0.05))
        
        ax.annotate('Canal de fluido\nconductivo', xy=(self.R, 0), 
                  xytext=(self.R, -0.5*self.a), 
                  arrowprops=dict(facecolor='black', shrink=0.05))
        
        # Añadir etiquetas y título
        ax.set_xlabel('Coordenada radial (r)')
        ax.set_ylabel('Coordenada axial (z)')
        ax.set_title(f'Sección transversal de la geometría pseudotoroidal en φ = {phi_value}')
        ax.grid(True)
        ax.axis('equal')
        
        return fig, ax


class EMSolver:
    """
    Resuelve las ecuaciones electromagnéticas en la geometría toroidal.
    Implementa la sección 2.2 del artículo para calcular los modos resonantes.
    """
    
    def __init__(self, geometry, material_props, frequency=5.64e9):
        """
        Inicializa el solucionador electromagnético.
        
        Args:
            geometry: Instancia de GeometryGenerator
            material_props: Propiedades del material superconductor
            frequency: Frecuencia de operación en Hz
        """
        self.geometry = geometry
        self.material = material_props
        self.frequency = frequency
        self.omega = 2 * np.pi * frequency
        
        # Calcular longitud de onda
        self.wavelength = Constants.SPEED_OF_LIGHT / frequency
        
        # Almacenamiento para los campos calculados
        self.E_field = None
        self.B_field = None
        self.energy_density = None
        
        logger.info(f"EMSolver inicializado con f={frequency/1e9} GHz, λ={self.wavelength} m")
    
    def calc_surface_resistance(self, temperature):
        """
        Calcula la resistencia superficial del superconductor según la ec. (12).
        
        Args:
            temperature: Temperatura en Kelvin
            
        Returns:
            Resistencia superficial en Ohms
        """
        if not hasattr(self.material, 'london_penetration_depth'):
            raise ValueError("El material debe tener definida london_penetration_depth")
        
        # Cálculo simplificado de la conductividad en estado normal a la temperatura dada
        # Utilizando una aproximación para la conductividad residual
        if temperature < self.material.critical_temperature:
            # Modelo de dos fluidos para T < Tc
            reduced_temp = temperature / self.material.critical_temperature
            sigma_n = self.material.conductivity * (reduced_temp**4)  # Aproximación
        else:
            sigma_n = self.material.conductivity
        
        # Ecuación (12) del artículo
        lambda_L = self.material.london_penetration_depth
        Rs = (self.omega**2 * mu_0**2 * lambda_L**3 * sigma_n) / 2
        
        return Rs
    
    def calc_quality_factor(self, temperature=20):
        """
        Calcula el factor Q para la cavidad según las ecuaciones (11) y (13).
        
        Args:
            temperature: Temperatura en Kelvin
            
        Returns:
            Factor Q cargado
        """
        # Calcular resistencia superficial
        Rs = self.calc_surface_resistance(temperature)
        
        # Estimar volumen y área superficial del toroide
        R = self.geometry.R
        a = self.geometry.a
        volume = 2 * np.pi**2 * R * a**2  # Volumen de un toro
        surface = 4 * np.pi**2 * R * a     # Área superficial de un toro
        
        # Ecuación (13) del artículo
        Q0 = (self.omega * mu_0 * self.material.london_penetration_depth * volume) / (Rs * surface)
        
        # Ecuación (11) del artículo
        beta_c = 0.1  # Coeficiente de acoplamiento típico
        Q_loaded = Q0 / (1 + beta_c)
        
        logger.info(f"Factor Q calculado: Q0={Q0:.1e}, Q_loaded={Q_loaded:.1e}")
        return Q_loaded
    
    def find_resonant_frequencies(self, n_range=(0, 5), m_range=(0, 5), k_range=(1, 5)):
        """
        Encuentra las frecuencias resonantes usando la ecuación (10) del artículo.
        
        Args:
            n_range: Rango de valores para índice n
            m_range: Rango de valores para índice m
            k_range: Rango de valores para índice k
            
        Returns:
            DataFrame con los modos resonantes y sus frecuencias
        """
        modes = []
        
        R = self.geometry.R
        a = self.geometry.a
        
        for n in range(n_range[0], n_range[1]):
            for m in range(m_range[0], m_range[1]):
                for k in range(k_range[0], k_range[1]):
                    # Estimación de la solución de la ecuación trascendental (10)
                    if n == 0 and m == 0:
                        # Modo TM_{0,1,k}
                        x_nk = 2.405 + (k-1) * np.pi  # Aproximación del k-ésimo cero de J_0
                    else:
                        # Modos de orden superior
                        x_nk = n + m/2 + (k-0.5) * np.pi
                        
                    omega_nmk = x_nk * Constants.SPEED_OF_LIGHT / np.sqrt(R*a)
                    freq_nmk = omega_nmk / (2*np.pi)
                    
                    # Calcular la longitud de onda
                    lambda_nmk = Constants.SPEED_OF_LIGHT / freq_nmk
                    
                    modes.append({
                        'n': n,
                        'm': m,
                        'k': k,
                        'frequency': freq_nmk,
                        'omega': omega_nmk,
                        'wavelength': lambda_nmk
                    })
        
        # Crear DataFrame y ordenar por frecuencia
        modes_df = pd.DataFrame(modes)
        modes_df = modes_df.sort_values('frequency')
        
        return modes_df
    
    def calc_field_distribution(self, n=1, m=0, k=1, resolution=30):
        """
        Calcula la distribución del campo electromagnético para un modo específico.
        Implementa la ecuación (9) y las relacionadas en la sección 2.2.1.
        
        Args:
            n, m, k: Índices del modo
            resolution: Resolución de la malla
            
        Returns:
            Arrays de componentes de campo eléctrico y magnético
        """
        # Crear malla en coordenadas toroidales
        eta = np.linspace(0.1, 2.0, resolution)  # Coordenada radial toroidal
        xi = np.linspace(0, 2*np.pi, resolution)  # Coordenada angular toroidal
        phi = np.linspace(0, 2*np.pi, resolution)  # Coordenada toroidal azimutal
        
        # Crear mallas 3D
        eta_grid, xi_grid, phi_grid = np.meshgrid(eta, xi, phi, indexing='ij')
        
        # Calcular la frecuencia resonante para este modo
        modes_df = self.find_resonant_frequencies(
            n_range=(n, n+1), 
            m_range=(m, m+1), 
            k_range=(k, k+1)
        )
        omega_nmk = modes_df.iloc[0]['omega']
        
        # Inicialización de componentes del campo eléctrico (simplificado)
        E_r = np.zeros_like(eta_grid)
        E_theta = np.zeros_like(eta_grid)
        E_phi = np.zeros_like(eta_grid)
        
        # Cálculo de las componentes del campo en cada punto de la malla
        for i in range(resolution):
            for j in range(resolution):
                for l in range(resolution):
                    try:
                        p_eta = lpmv(m, n, np.cos(eta_grid[i,j,l]))
                        p_xi = lpmv(m, n, np.cos(xi_grid[i,j,l]))
                    except:
                        p_eta = 0
                        p_xi = 0
                    
                    # Componente φ del campo eléctrico (modo TE dominante)
                    E_phi[i,j,l] = p_eta * p_xi * np.sin(m * phi_grid[i,j,l])
                    
                    if n > 0:
                        dp_eta = -m * np.sin(eta_grid[i,j,l]) * lpmv(m, n-1, np.cos(eta_grid[i,j,l]))
                        dp_xi = -m * np.sin(xi_grid[i,j,l]) * lpmv(m, n-1, np.cos(xi_grid[i,j,l]))
                        
                        E_r[i,j,l] = dp_eta * p_xi * np.cos(m * phi_grid[i,j,l])
                        E_theta[i,j,l] = p_eta * dp_xi * np.cos(m * phi_grid[i,j,l])
        
        # Normalización de los campos
        E_max = np.sqrt(E_r**2 + E_theta**2 + E_phi**2).max()
        if E_max > 0:
            E_r /= E_max
            E_theta /= E_max
            E_phi /= E_max
        
        # Aproximación para el campo magnético a partir de E
        B_r = -E_phi / Constants.SPEED_OF_LIGHT
        B_theta = E_r / Constants.SPEED_OF_LIGHT
        B_phi = -E_theta / Constants.SPEED_OF_LIGHT
        
        energy_density = (epsilon_0 * (E_r**2 + E_theta**2 + E_phi**2) + 
                          (1/mu_0) * (B_r**2 + B_theta**2 + B_phi**2)) / 2
        
        a = self.geometry.a
        R = self.geometry.R
        
        x = (R + a * np.cos(xi_grid) * np.cos(eta_grid)) * np.cos(phi_grid)
        y = (R + a * np.cos(xi_grid) * np.cos(eta_grid)) * np.sin(phi_grid)
        z = a * np.cos(xi_grid) * np.sin(eta_grid)
        
        self.E_field = {'r': E_r, 'theta': E_theta, 'phi': E_phi, 
                        'x': x, 'y': y, 'z': z, 'magnitude': np.sqrt(E_r**2 + E_theta**2 + E_phi**2)}
        self.B_field = {'r': B_r, 'theta': B_theta, 'phi': B_phi,
                        'x': x, 'y': y, 'z': z, 'magnitude': np.sqrt(B_r**2 + B_theta**2 + B_phi**2)}
        self.energy_density = {'density': energy_density, 'x': x, 'y': y, 'z': z}
        
        return self.E_field, self.B_field, self.energy_density
    
    def visualize_fields(self, field_type='E', component='magnitude', slice_dim='z', slice_pos=0):
        """
        Visualiza la distribución de campo eléctrico o magnético en un corte.
        
        Args:
            field_type: 'E' para campo eléctrico, 'B' para campo magnético
            component: 'r', 'theta', 'phi', o 'magnitude'
            slice_dim: Dimensión para el corte ('x', 'y', o 'z')
            slice_pos: Posición relativa del corte (entre 0 y 1)
            
        Returns:
            Figure y axes de matplotlib con la visualización
        """
        if field_type == 'E' and self.E_field is None:
            raise ValueError("Debe calcular la distribución de campo primero con calc_field_distribution()")
        elif field_type == 'B' and self.B_field is None:
            raise ValueError("Debe calcular la distribución de campo primero con calc_field_distribution()")
        
        field = self.E_field if field_type == 'E' else self.B_field
        
        x = field['x']
        y = field['y']
        z = field['z']
        
        if component in ['r', 'theta', 'phi', 'magnitude']:
            field_comp = field[component]
        else:
            raise ValueError("Componente no válida. Use 'r', 'theta', 'phi', o 'magnitude'")
        
        if slice_dim == 'x':
            idx = np.argmin(np.abs(x - slice_pos * np.max(x)))
            x_slice = y[:, idx, :]
            y_slice = z[:, idx, :]
            field_slice = field_comp[:, idx, :]
            xlabel, ylabel = 'Y', 'Z'
        elif slice_dim == 'y':
            idx = np.argmin(np.abs(y - slice_pos * np.max(y)))
            x_slice = x[:, :, idx]
            y_slice = z[:, :, idx]
            field_slice = field_comp[:, :, idx]
            xlabel, ylabel = 'X', 'Z'
        elif slice_dim == 'z':
            idx = np.argmin(np.abs(z - slice_pos * np.max(z)))
            x_slice = x[idx, :, :]
            y_slice = y[idx, :, :]
            field_slice = field_comp[idx, :, :]
            xlabel, ylabel = 'X', 'Y'
        else:
            raise ValueError("Dimensión de corte no válida. Use 'x', 'y', o 'z'")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        cf = ax.contourf(x_slice, y_slice, field_slice, 50, cmap='plasma')
        
        if component != 'magnitude':
            step = 5
            if field_type == 'E':
                if slice_dim == 'x':
                    u = field['theta'][::step, idx, ::step]
                    v = field['phi'][::step, idx, ::step]
                elif slice_dim == 'y':
                    u = field['r'][::step, ::step, idx]
                    v = field['phi'][::step, ::step, idx]
                else:  # slice_dim == 'z'
                    u = field['r'][idx, ::step, ::step]
                    v = field['theta'][idx, ::step, ::step]
            else:  # field_type == 'B'
                if slice_dim == 'x':
                    u = field['theta'][::step, idx, ::step]
                    v = field['phi'][::step, idx, ::step]
                elif slice_dim == 'y':
                    u = field['r'][::step, ::step, idx]
                    v = field['phi'][::step, ::step, idx]
                else:
                    u = field['r'][idx, ::step, ::step]
                    v = field['theta'][idx, ::step, ::step]
            
            x_quiver = x_slice[::step, ::step]
            y_quiver = y_slice[::step, ::step]
            magnitude = np.sqrt(u**2 + v**2)
            if magnitude.max() > 0:
                u = u / magnitude.max()
                v = v / magnitude.max()
            
            ax.quiver(x_quiver, y_quiver, u, v, angles='xy', scale=30, color='white', alpha=0.7)
        
        cbar = fig.colorbar(cf, ax=ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        field_name = "Eléctrico" if field_type == 'E' else "Magnético"
        component_name = {
            'r': 'radial',
            'theta': 'poloidal',
            'phi': 'toroidal',
            'magnitude': 'magnitud'
        }.get(component, component)
        
        ax.set_title(f"Campo {field_name} - Componente {component_name} (Corte {slice_dim.upper()})")
        ax.set_aspect('equal')
        
        return fig, ax
    
    def visualize_energy_density(self, slice_dim='z', slice_pos=0):
        """
        Visualiza la distribución de densidad de energía electromagnética en un corte.
        
        Args:
            slice_dim: Dimensión para el corte ('x', 'y', o 'z')
            slice_pos: Posición relativa del corte (entre 0 y 1)
            
        Returns:
            Figure y axes de matplotlib con la visualización
        """
        if self.energy_density is None:
            raise ValueError("Debe calcular la distribución de campo primero con calc_field_distribution()")
        
        x = self.energy_density['x']
        y = self.energy_density['y']
        z = self.energy_density['z']
        energy = self.energy_density['density']
        
        if slice_dim == 'x':
            idx = np.argmin(np.abs(x - slice_pos * np.max(x)))
            x_slice = y[:, idx, :]
            y_slice = z[:, idx, :]
            energy_slice = energy[:, idx, :]
            xlabel, ylabel = 'Y', 'Z'
        elif slice_dim == 'y':
            idx = np.argmin(np.abs(y - slice_pos * np.max(y)))
            x_slice = x[:, :, idx]
            y_slice = z[:, :, idx]
            energy_slice = energy[:, :, idx]
            xlabel, ylabel = 'X', 'Z'
        elif slice_dim == 'z':
            idx = np.argmin(np.abs(z - slice_pos * np.max(z)))
            x_slice = x[idx, :, :]
            y_slice = y[idx, :, :]
            energy_slice = energy[idx, :, :]
            xlabel, ylabel = 'X', 'Y'
        else:
            raise ValueError("Dimensión de corte no válida. Use 'x', 'y', o 'z'")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        cf = ax.contourf(x_slice, y_slice, energy_slice, 50, cmap='inferno')
        contour = ax.contour(x_slice, y_slice, energy_slice, 10, colors='white', alpha=0.3)
        ax.clabel(contour, inline=True, fontsize=8, fmt='%.1e')
        
        cbar = fig.colorbar(cf, ax=ax)
        cbar.set_label('Densidad de energía (J/m³)')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Densidad de energía electromagnética (Corte {slice_dim.upper()})")
        ax.set_aspect('equal')
        
        return fig, ax
    
    def visualize_3d_field(self, field_type='E', component='magnitude', threshold=0.5):
        """
        Crea una visualización 3D de los campos electromagnéticos.
        
        Args:
            field_type: 'E' para campo eléctrico, 'B' para campo magnético, 'energy' para densidad de energía
            component: 'r', 'theta', 'phi', o 'magnitude' (o 'density' para energía)
            threshold: Umbral para mostrar puntos (valor entre 0 y 1)
            
        Returns:
            Figure y axes de matplotlib con la visualización 3D
        """
        if (field_type in ['E', 'B'] and self.E_field is None) or (field_type == 'energy' and self.energy_density is None):
            raise ValueError("Debe calcular la distribución de campo primero con calc_field_distribution()")
        
        if field_type == 'E':
            field = self.E_field
        elif field_type == 'B':
            field = self.B_field
        else:
            field = self.energy_density
            component = 'density'
        
        x = field['x']
        y = field['y']
        z = field['z']
        
        if component in field:
            values = field[component]
        else:
            raise ValueError(f"Componente '{component}' no válida para el campo {field_type}")
        
        normalized_values = values / np.max(values)
        mask = normalized_values > threshold
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        step = 3
        scatter = ax.scatter(
            x[mask][::step], 
            y[mask][::step], 
            z[mask][::step],
            c=values[mask][::step],
            cmap='plasma',
            alpha=0.8,
            s=20,
        )
        
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.5)
        
        if field_type == 'E':
            field_label = "Campo Eléctrico"
        elif field_type == 'B':
            field_label = "Campo Magnético"
        else:
            field_label = "Densidad de Energía"
        
        component_label = {
            'r': 'Componente Radial',
            'theta': 'Componente Poloidal',
            'phi': 'Componente Toroidal',
            'magnitude': 'Magnitud',
            'density': 'Densidad'
        }.get(component, component)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"{field_label} - {component_label}")
        ax.set_xlim([x.min(), x.max()])
        ax.set_ylim([y.min(), y.max()])
        ax.set_zlim([z.min(), z.max()])
        
        return fig, ax


class AcousticSolver:
    """
    Resuelve las ecuaciones acústicas en la geometría toroidal.
    Implementa la sección 2.2.3 del artículo para modos acústicos y acoplamiento cruzado.
    """
    
    def __init__(self, geometry, fluid_props):
        """
        Inicializa el solucionador acústico.
        
        Args:
            geometry: Instancia de GeometryGenerator
            fluid_props: Propiedades del fluido conductivo
        """
        self.geometry = geometry
        self.fluid = fluid_props
        
        if not hasattr(self.fluid, 'speed_of_sound') or self.fluid.speed_of_sound == 0:
            raise ValueError("El fluido debe tener definida la velocidad del sonido")
        
        self.pressure_field = None
        self.velocity_field = None
        
        logger.info(f"AcousticSolver inicializado con c_s={self.fluid.speed_of_sound} m/s")
    
    def find_acoustic_modes(self, n_range=(0, 5), l_range=(1, 5), m_range=(0, 5)):
        """
        Encuentra los modos acústicos según la ecuación (15) del artículo.
        
        Args:
            n_range: Rango para el índice n
            l_range: Rango para el índice l
            m_range: Rango para el índice m
            
        Returns:
            DataFrame con los modos acústicos y sus frecuencias
        """
        modes = []
        
        R = self.geometry.R
        a = self.geometry.a
        c_s = self.fluid.speed_of_sound
        
        for n in range(n_range[0], n_range[1]):
            for l in range(l_range[0], l_range[1]):
                for m in range(m_range[0], m_range[1]):
                    try:
                        if n == 0:
                            X_nl = jv(n, l*np.pi)
                        else:
                            X_nl = (l + n/2 - 0.25) * np.pi
                    except:
                        X_nl = l * np.pi
                        
                    omega_nlm = c_s * np.sqrt((X_nl**2)/(a**2) + (m**2)/(R**2))
                    freq_nlm = omega_nlm / (2*np.pi)
                    
                    modes.append({
                        'n': n,
                        'l': l,
                        'm': m,
                        'X_nl': X_nl,
                        'frequency': freq_nlm,
                        'omega': omega_nlm
                    })
        
        modes_df = pd.DataFrame(modes)
        modes_df = modes_df.sort_values('frequency')
        
        return modes_df
    
    def find_cross_domain_resonances(self, em_solver, threshold=0.05):
        """
        Identifica resonancias cruzadas entre dominios electromagnético y acústico
        según la ecuación (16) del artículo.
        
        Args:
            em_solver: Instancia del EMSolver con modos calculados
            threshold: Tolerancia relativa
            
        Returns:
            DataFrame con las resonancias cruzadas identificadas
        """
        em_modes = em_solver.find_resonant_frequencies()
        acoustic_modes = self.find_acoustic_modes()
        
        cross_resonances = []
        
        for _, em_mode in em_modes.iterrows():
            for _, ac_mode in acoustic_modes.iterrows():
                ratio = ac_mode['omega'] / (em_mode['omega']/2)
                
                if abs(ratio - 1) < threshold:
                    cross_resonances.append({
                        'em_n': em_mode['n'],
                        'em_m': em_mode['m'],
                        'em_k': em_mode['k'],
                        'em_freq': em_mode['frequency'],
                        'acoustic_n': ac_mode['n'],
                        'acoustic_l': ac_mode['l'],
                        'acoustic_m': ac_mode['m'],
                        'acoustic_freq': ac_mode['frequency'],
                        'ratio': ratio,
                        'difference': abs(ratio - 1)
                    })
        
        if cross_resonances:
            resonances_df = pd.DataFrame(cross_resonances)
            resonances_df = resonances_df.sort_values('difference')
        else:
            resonances_df = pd.DataFrame(columns=[
                'em_n', 'em_m', 'em_k', 'em_freq',
                'acoustic_n', 'acoustic_l', 'acoustic_m', 'acoustic_freq',
                'ratio', 'difference'
            ])
            
        return resonances_df
    
    def calc_pressure_distribution(self, n=0, l=1, m=0, amplitude=1.0, resolution=30):
        """
        Calcula la distribución de presión acústica para un modo específico.
        
        Args:
            n, l, m: Índices del modo acústico
            amplitude: Amplitud de la presión
            resolution: Resolución de la malla
            
        Returns:
            Arrays de presión acústica y campos de velocidad
        """
        r = np.linspace(0, self.geometry.a, resolution)
        theta = np.linspace(0, 2*np.pi, resolution)
        phi = np.linspace(0, 2*np.pi, resolution)
        
        r_grid, theta_grid, phi_grid = np.meshgrid(r, theta, phi, indexing='ij')
        
        R = self.geometry.R
        a = self.geometry.a
        
        modes_df = self.find_acoustic_modes(
            n_range=(n, n+1), 
            l_range=(l, l+1), 
            m_range=(m, m+1)
        )
        X_nl = modes_df.iloc[0]['X_nl']
        omega = modes_df.iloc[0]['omega']
        
        bessel_term = jv(n, X_nl * r_grid / a)
        theta_term = np.cos(n * theta_grid)
        phi_term = np.cos(m * phi_grid)
        
        p_acoustic = amplitude * bessel_term * theta_term * phi_term
        
        dr = r[1] - r[0] if resolution > 1 else 1
        dtheta = theta[1] - theta[0] if resolution > 1 else 1
        dphi = phi[1] - phi[0] if resolution > 1 else 1
        
        v_r = np.zeros_like(p_acoustic)
        v_theta = np.zeros_like(p_acoustic)
        v_phi = np.zeros_like(p_acoustic)
        
        for i in range(1, resolution-1):
            for j in range(1, resolution-1):
                for k in range(1, resolution-1):
                    v_r[i,j,k] = -(p_acoustic[i+1,j,k] - p_acoustic[i-1,j,k]) / (2*dr)
                    v_theta[i,j,k] = -(p_acoustic[i,j+1,k] - p_acoustic[i,j-1,k]) / (2*dtheta)
                    v_phi[i,j,k] = -(p_acoustic[i,j,k+1] - p_acoustic[i,j,k-1]) / (2*dphi)
        
        v_max = np.sqrt(v_r**2 + v_theta**2 + v_phi**2).max()
        if v_max > 0:
            v_r /= v_max
            v_theta /= v_max
            v_phi /= v_max
        
        x = (R + r_grid * np.cos(theta_grid)) * np.cos(phi_grid)
        y = (R + r_grid * np.cos(theta_grid)) * np.sin(phi_grid)
        z = r_grid * np.sin(theta_grid)
        
        self.pressure_field = {
            'pressure': p_acoustic,
            'x': x, 'y': y, 'z': z,
            'mode': (n, l, m),
            'omega': omega
        }
        
        self.velocity_field = {
            'r': v_r, 'theta': v_theta, 'phi': v_phi,
            'x': x, 'y': y, 'z': z
        }
        
        return self.pressure_field, self.velocity_field
    
    def visualize_pressure(self, slice_dim='z', slice_pos=0):
        """
        Visualiza la distribución de presión acústica en un corte.
        
        Args:
            slice_dim: Dimensión para el corte ('x', 'y', o 'z')
            slice_pos: Posición relativa del corte (entre 0 y 1)
            
        Returns:
            Figure y axes de matplotlib con la visualización
        """
        if self.pressure_field is None:
            raise ValueError("Debe calcular la distribución de presión primero con calc_pressure_distribution()")
        
        x = self.pressure_field['x']
        y = self.pressure_field['y']
        z = self.pressure_field['z']
        pressure = self.pressure_field['pressure']
        
        if slice_dim == 'x':
            idx = np.argmin(np.abs(x - slice_pos * np.max(x)))
            x_slice = y[:, idx, :]
            y_slice = z[:, idx, :]
            pressure_slice = pressure[:, idx, :]
            xlabel, ylabel = 'Y', 'Z'
        elif slice_dim == 'y':
            idx = np.argmin(np.abs(y - slice_pos * np.max(y)))
            x_slice = x[:, :, idx]
            y_slice = z[:, :, idx]
            pressure_slice = pressure[:, :, idx]
            xlabel, ylabel = 'X', 'Z'
        elif slice_dim == 'z':
            idx = np.argmin(np.abs(z - slice_pos * np.max(z)))
            x_slice = x[idx, :, :]
            y_slice = y[idx, :, :]
            pressure_slice = pressure[idx, :, :]
            xlabel, ylabel = 'X', 'Y'
        else:
            raise ValueError("Dimensión de corte no válida. Use 'x', 'y', o 'z'")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        cf = ax.contourf(x_slice, y_slice, pressure_slice, 50, cmap='coolwarm')
        contour = ax.contour(x_slice, y_slice, pressure_slice, 10, colors='black', alpha=0.3)
        ax.clabel(contour, inline=True, fontsize=8)
        
        cbar = fig.colorbar(cf, ax=ax)
        cbar.set_label('Presión acústica (relativa)')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        n, l, m = self.pressure_field['mode']
        freq = self.pressure_field['omega'] / (2*np.pi)
        
        ax.set_title(f"Presión acústica - Modo ({n},{l},{m}) - {freq:.2f} Hz")
        ax.set_aspect('equal')
        
        return fig, ax


class FluidDynamicsSolver:
    """
    Resuelve las ecuaciones de dinámica de fluidos y MHD en la geometría toroidal.
    Implementa las ecuaciones de la sección 2.3 del artículo.
    """
    
    def __init__(self, geometry, fluid_props, B_field=1.0):
        """
        Inicializa el solucionador de dinámica de fluidos.
        
        Args:
            geometry: Instancia de GeometryGenerator
            fluid_props: Propiedades del fluido conductivo
            B_field: Intensidad del campo magnético externo (T)
        """
        self.geometry = geometry
        self.fluid = fluid_props
        self.B_field = B_field
        self.velocity_field = None
        logger.info(f"FluidDynamicsSolver inicializado con B_field={B_field} T")
    
    def compute_mhd_flow(self, resolution=50):
        """
        Calcula un campo de velocidad MHD dummy (placeholder) en la geometría toroidal.
        
        Args:
            resolution: Resolución de la malla
            
        Returns:
            Diccionario con coordenadas y componentes del campo de velocidad
        """
        theta_vals = np.linspace(0, 2*np.pi, resolution)
        phi_vals = np.linspace(0, 2*np.pi, resolution)
        theta_grid, phi_grid = np.meshgrid(theta_vals, phi_vals)
        
        # Obtener posición usando el toroide estándar
        x, y, z = self.geometry.standard_torus(theta_grid, phi_grid)
        
        # Campo de velocidad dummy influenciado por B_field
        v_theta = self.B_field * np.sin(theta_grid)
        v_phi = self.B_field * np.cos(phi_grid)
        
        self.velocity_field = {'x': x, 'y': y, 'z': z, 'v_theta': v_theta, 'v_phi': v_phi}
        logger.info("Campo de velocidad MHD calculado (placeholder).")
        return self.velocity_field
      
    def visualize_flow(self, component='v_theta', step=2):
        """
        Visualiza el campo de velocidad MHD en una proyección 2D.
        
        Args:
            component: Componente a visualizar ('v_theta' o 'v_phi')
            step: Factor de reducción de puntos para la visualización
            
        Returns:
            Figure y axes de matplotlib con la visualización
        """
        if self.velocity_field is None:
            raise ValueError("Debe calcular el flujo primero con compute_mhd_flow()")
            
        x = self.velocity_field['x']
        y = self.velocity_field['y']
        v_comp = self.velocity_field[component]
        
        fig, ax = plt.subplots(figsize=(10,8))
        ax.quiver(x[::step, ::step], y[::step, ::step], v_comp[::step, ::step], v_comp[::step, ::step],
                  angles='xy', scale=30, color='blue', alpha=0.8)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"Visualización del flujo MHD - Componente {component}")
        ax.set_aspect('equal')
        return fig, ax


def main():
    # Crear geometría
    geom = GeometryGenerator(R=1.0, a=0.3, r0=0.2, z0=0.0, alpha=1.0, beta=10.0, epsilon=0.05, Rt=0.7)
    
    # Visualizar geometría pseudotoroidal
    fig_geom, ax_geom = geom.visualize_geometry()
    plt.show()
    
    # Cargar propiedades de materiales y fluidos
    materials = MaterialProperties.load_material_library()
    superconductor = materials["Nb"]
    fluid = materials["Mercury"]
    
    # Solucionador electromagnético
    em_solver = EMSolver(geom, superconductor, frequency=5.64e9)
    E_field, B_field, energy_density = em_solver.calc_field_distribution(n=1, m=0, k=1, resolution=30)
    fig_E, ax_E = em_solver.visualize_fields(field_type='E', component='magnitude', slice_dim='z', slice_pos=0.5)
    plt.show()
    
    # Solucionador acústico
    acoustic_solver = AcousticSolver(geom, fluid)
    pressure_field, velocity_field = acoustic_solver.calc_pressure_distribution(n=0, l=1, m=0, amplitude=1.0, resolution=30)
    fig_pressure, ax_pressure = acoustic_solver.visualize_pressure(slice_dim='z', slice_pos=0.5)
    plt.show()
    
    # Solucionador de dinámica de fluidos/MHD
    fluid_solver = FluidDynamicsSolver(geom, fluid, B_field=1.0)
    fluid_solver.compute_mhd_flow(resolution=50)
    fig_flow, ax_flow = fluid_solver.visualize_flow(component='v_theta', step=2)
    plt.show()


if __name__ == "__main__":
    main()
