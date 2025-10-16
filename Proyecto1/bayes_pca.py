import cv2
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

class ClasificadorPCA:
    def __init__(self, n_components=None):
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()
        self.mean_class0 = None
        self.mean_class1 = None
        self.cov_class0 = None
        self.cov_class1 = None
        self.prior_class0 = 0.5
        self.prior_class1 = 0.5
        self.optimal_threshold = None
        self.optimal_components = None
        
    def cargar_imagenes(self, dataset, ruta_base, conjunto='train'):
        """Carga imágenes y máscaras desde el dataset"""
        images = []
        masks = []
        
        for img_file, mask_file in tqdm(zip(dataset[conjunto]['images'], dataset[conjunto]['masks']), 
                                       desc=f"Cargando {conjunto}"):
            try:
                # Cargar imagen (tamaño original)
                img_path = Path(ruta_base) / img_file
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Cargar máscara en su tamaño original
                mask_path = Path(ruta_base) / mask_file
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    continue
                
                # Asegurar que máscara tenga las mismas dimensiones que la imagen
                if img.shape[:2] != mask.shape:
                    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
                
                mask = (mask > 128).astype(np.uint8)
                
                images.append(img)
                masks.append(mask)
                
            except Exception as e:
                continue
        
        return images, masks
    
    def extraer_caracteristicas_pixel(self, images, masks, max_pixels_per_image=1000):
        """Extrae características a nivel de píxel"""
        X_pixels = []
        y_pixels = []
        
        for img, mask in tqdm(zip(images, masks), desc="Extrayendo características"):
            try:
                height, width = img.shape[:2]
                
                if mask.shape != (height, width):
                    continue
                
                # Muestrear píxeles de lesión
                lesion_rows, lesion_cols = np.where(mask > 0)
                n_lesion = len(lesion_rows)
                
                # Muestrear píxeles de no lesión
                background_rows, background_cols = np.where(mask == 0)
                n_background = len(background_rows)
                
                n_samples = min(max_pixels_per_image // 2, n_lesion, n_background)
                
                if n_samples > 0:
                    # Píxeles de lesión
                    indices_lesion = np.random.choice(n_lesion, n_samples, replace=False)
                    for idx in indices_lesion:
                        r, c = lesion_rows[idx], lesion_cols[idx]
                        features = self._extraer_caracteristicas_pixel_individual(img, r, c)
                        X_pixels.append(features)
                        y_pixels.append(1)
                    
                    # Píxeles de no lesión
                    indices_background = np.random.choice(n_background, n_samples, replace=False)
                    for idx in indices_background:
                        r, c = background_rows[idx], background_cols[idx]
                        features = self._extraer_caracteristicas_pixel_individual(img, r, c)
                        X_pixels.append(features)
                        y_pixels.append(0)
                        
            except Exception as e:
                continue
        
        return np.array(X_pixels), np.array(y_pixels)
    
    def _extraer_caracteristicas_pixel_individual(self, img, row, col, window_size=5):
        """Extrae características para un píxel individual"""
        features = []
        height, width = img.shape[:2]
        
        # Características del píxel central
        pixel_rgb = img[row, col]
        features.extend(pixel_rgb)
        
        # Convertir a HSV
        pixel_hsv = cv2.cvtColor(np.uint8([[pixel_rgb]]), cv2.COLOR_RGB2HSV)[0][0]
        features.extend(pixel_hsv)
        
        # Vecindario
        half_window = window_size // 2
        r_start = max(0, row - half_window)
        r_end = min(height, row + half_window + 1)
        c_start = max(0, col - half_window)
        c_end = min(width, col + half_window + 1)
        
        neighborhood = img[r_start:r_end, c_start:c_end]
        
        # Estadísticas del vecindario
        for channel in range(3):
            channel_data = neighborhood[:, :, channel].flatten()
            features.extend([
                np.mean(channel_data),
                np.std(channel_data),
                np.median(channel_data)
            ])
        
        return features
    
    def entrenar_con_validacion_externa(self, X_train, y_train, X_val, y_val):
        """Método simple sin parámetros opcionales"""
        return self._entrenar_modelo_pixel(X_train, y_train, X_val, y_val)
    
    def _entrenar_modelo_pixel(self, X_train_pixels, y_train_pixels, X_val_pixels, y_val_pixels):
        """
        Entrena el modelo usando SOLO training para entrenamiento
        y validation SOLO para validación y selección de hiperparámetros
        """
        print(f"Píxeles training: {len(X_train_pixels)}")
        print(f"Píxeles validation: {len(X_val_pixels)}")

        # 1. PREPROCESAMIENTO (solo con training)
        X_train_std = self.scaler.fit_transform(X_train_pixels)
        X_val_std = self.scaler.transform(X_val_pixels)  # Transformar validation con mismo scaler

        # 2. PCA (solo con training)
        X_train_pca = self.pca.fit_transform(X_train_std)
        X_val_pca = self.pca.transform(X_val_std)  # Transformar validation con mismo PCA

        # 3. SELECCIÓN DE COMPONENTES (usando validation)
        self.optimal_components, best_youden = self._seleccionar_componentes_optimas(
            X_train_pca, y_train_pixels, X_val_pca, y_val_pixels
        )

        print(f"Componentes óptimas: {self.optimal_components}, Youden: {best_youden:.3f}")

        # 4. ENTRENAMIENTO FINAL (Conjunto de entrenamiento)
        X_train_opt = X_train_pca[:, :self.optimal_components]
        self._entrenar_clasificador_bayesiano(X_train_opt, y_train_pixels)

        # 5. SELECCIÓN DE UMBRAL (Conjunto de validación)
        X_val_opt = X_val_pca[:, :self.optimal_components]
        self._encontrar_umbral_optimal(X_val_opt, y_val_pixels)

        return best_youden
    
    def _seleccionar_componentes_optimas(self, X_train_pca, y_train, X_val_pca, y_val):
        """Selecciona componentes óptimas usando validation para evaluación"""
        best_youden = -1
        best_components = 1
        
        for n_comp in range(1, min(25, X_train_pca.shape[1]) + 1):
            try:
                # Entrenar con TRAINING
                X_train_reduced = X_train_pca[:, :n_comp]
                self._entrenar_clasificador_bayesiano(X_train_reduced, y_train)
                
                # Evaluar con VALIDATION
                X_val_reduced = X_val_pca[:, :n_comp]
                y_proba = self._calcular_probabilidades(X_val_reduced)
                youden = self._calcular_indice_youden(y_val, y_proba)
                
                if youden > best_youden:
                    best_youden = youden
                    best_components = n_comp
                    
            except Exception as e:
                continue
        
        return best_components, best_youden
    
    def _entrenar_clasificador_bayesiano(self, features, labels):
        """Entrena el clasificador bayesiano"""
        class0_features = features[labels == 0]
        class1_features = features[labels == 1]
        
        self.mean_class0 = np.mean(class0_features, axis=0)
        self.mean_class1 = np.mean(class1_features, axis=0)
        self.cov_class0 = np.cov(class0_features, rowvar=False)
        self.cov_class1 = np.cov(class1_features, rowvar=False)
        
        self.prior_class0 = len(class0_features) / len(features)
        self.prior_class1 = len(class1_features) / len(features)
    
    def _encontrar_umbral_optimal(self, X_val, y_val):
        """Encuentra umbral óptimo usando validation"""
        y_proba = self._calcular_probabilidades(X_val)
        fpr, tpr, thresholds = roc_curve(y_val, y_proba)
        youden_scores = tpr - fpr
        optimal_idx = np.argmax(youden_scores)
        self.optimal_threshold = thresholds[optimal_idx]
    
    def _calcular_probabilidades(self, features):
        """Calcula probabilidades posteriores"""
        likelihood_class0 = multivariate_normal.pdf(
            features, mean=self.mean_class0, cov=self.cov_class0, allow_singular=True
        )
        likelihood_class1 = multivariate_normal.pdf(
            features, mean=self.mean_class1, cov=self.cov_class1, allow_singular=True
        )
        
        evidence = likelihood_class0 * self.prior_class0 + likelihood_class1 * self.prior_class1
        return (likelihood_class1 * self.prior_class1) / np.maximum(evidence, 1e-10)
    
    def _calcular_indice_youden(self, y_true, y_proba):
        """Calcula Índice de Youden"""
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        return np.max(tpr - fpr)

def visualizar_pca(clasificador, X_train, y_train, X_val, y_val):
    """
    Visualiza la varianza explicada por las componentes PCA
    """
    # Estandarizar y aplicar PCA
    X_train_std = clasificador.scaler.transform(X_train)
    X_val_std = clasificador.scaler.transform(X_val)
    X_train_pca = clasificador.pca.transform(X_train_std)
    
    # Varianza explicada
    explained_variance = clasificador.pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    plt.figure(figsize=(15, 5))
    
    # Gráfico 1: Varianza explicada individual
    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.6, color='skyblue')
    plt.xlabel('Componente Principal')
    plt.ylabel('Varianza Explicada')
    plt.title('Varianza Explicada por Componente PCA')
    plt.axvline(x=clasificador.optimal_components, color='red', linestyle='--', 
                label=f'Óptimo: {clasificador.optimal_components} componentes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Gráfico 2: Varianza acumulada
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'o-', color='orange', linewidth=2)
    plt.xlabel('Número de Componentes')
    plt.ylabel('Varianza Acumulada Explicada')
    plt.title('Varianza Acumulada PCA')
    plt.axhline(y=0.95, color='green', linestyle='--', label='95% de varianza')
    plt.axvline(x=clasificador.optimal_components, color='red', linestyle='--',
                label=f'Óptimo: {clasificador.optimal_components} componentes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Varianza explicada por las primeras {clasificador.optimal_components} componentes: {cumulative_variance[clasificador.optimal_components-1]:.3f}")

def visualizar_auc_y_matriz_confusion(clasificador, X_val, y_val):
    """
    Visualiza curva ROC/AUC y matriz de confusión
    """
    # Preprocesar datos de validación
    X_val_std = clasificador.scaler.transform(X_val)
    X_val_pca = clasificador.pca.transform(X_val_std)
    X_val_opt = X_val_pca[:, :clasificador.optimal_components]
    
    # Predecir probabilidades
    y_proba = clasificador._calcular_probabilidades(X_val_opt)
    y_pred = (y_proba >= clasificador.optimal_threshold).astype(int)
    
    # Calcular métricas
    fpr, tpr, thresholds = roc_curve(y_val, y_proba)
    roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(y_val, y_pred)
    
    # Crear figura con subplots
    plt.figure(figsize=(15, 6))
    
    # Gráfico 1: Curva ROC
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Clasificador aleatorio')
    
    # Marcar punto óptimo (Youden)
    youden_scores = tpr - fpr
    optimal_idx = np.argmax(youden_scores)
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', s=100, 
                label=f'Umbral óptimo (Youden = {youden_scores[optimal_idx]:.3f})')
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad)')
    plt.ylabel('Tasa de Verdaderos Positivos (Sensibilidad)')
    plt.title('Curva ROC - Clasificador Bayesiano')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # Gráfico 2: Matriz de Confusión
    plt.subplot(1, 2, 2)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No lesión', 'Lesión'])
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Matriz de Confusión\n(Conjunto de Validación)')
    
    # Añadir métricas a la matriz de confusión
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    plt.text(0.4, -0.3, f'Accuracy: {accuracy:.3f}\nSensibilidad: {sensitivity:.3f}\nEspecificidad: {specificity:.3f}\nPrecisión: {precision:.3f}', 
             transform=plt.gca().transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.tight_layout()
    plt.show()
    
    # Imprimir métricas detalladas
    print("\n" + "="*60)
    print("MÉTRICAS DETALLADAS - CONJUNTO DE VALIDACIÓN")
    print("="*60)
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Sensibilidad: {sensitivity:.4f} (Capacidad de detectar lesiones)")
    print(f"Especificidad: {specificity:.4f} (Capacidad de evitar falsos positivos)")
    print(f"Precisión:    {precision:.4f} (Lesiones correctamente identificadas)")
    print(f"Índice de Youden: {youden_scores[optimal_idx]:.4f}")
    print(f"Umbral óptimo: {clasificador.optimal_threshold:.4f}")
    print(f"AUC:         {roc_auc:.4f}")
    
    return roc_auc, cm

def evaluar_en_test(clasificador, test_images, test_masks):
    """
    Evalúa el modelo en el conjunto de test
    """
    print("\n" + "="*60)
    print("EVALUACIÓN EN CONJUNTO DE TEST")
    print("="*60)
    
    # Extraer características de test
    X_test, y_test = clasificador.extraer_caracteristicas_pixel(test_images, test_masks, max_pixels_per_image=1000)
    print(f"Píxeles de test: {len(X_test)}")
    print(f"Distribución: {np.bincount(y_test)}")
    
    # Preprocesar
    X_test_std = clasificador.scaler.transform(X_test)
    X_test_pca = clasificador.pca.transform(X_test_std)
    X_test_opt = X_test_pca[:, :clasificador.optimal_components]
    
    # Predecir
    y_proba_test = clasificador._calcular_probabilidades(X_test_opt)
    y_pred_test = (y_proba_test >= clasificador.optimal_threshold).astype(int)
    
    # Calcular métricas
    cm_test = confusion_matrix(y_test, y_pred_test)
    tn, fp, fn, tp = cm_test.ravel()
    
    accuracy_test = (tp + tn) / (tp + tn + fp + fn)
    sensitivity_test = tp / (tp + fn)
    specificity_test = tn / (tn + fp)
    
    print(f"Accuracy (Test):    {accuracy_test:.4f}")
    print(f"Sensibilidad (Test): {sensitivity_test:.4f}")
    print(f"Especificidad (Test): {specificity_test:.4f}")
    
    return X_test, y_test, y_proba_test, y_pred_test