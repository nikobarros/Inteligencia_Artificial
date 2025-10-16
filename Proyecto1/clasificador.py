import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix

dataset_ruta = "./dataset"

def cargar_imagenes_dermastocopia(ruta):
    path = Path(ruta)
    files = sorted([f.name for f in path.iterdir() if f.is_file()])
    
    imagenes_binarias = [f for f in files if '_expert' in f and f.endswith('.png')]
    imagenes_originales = [f for f in files if '_expert' not in f and f.endswith('.jpg')]
    
    print(f"Se han cargado {len(imagenes_originales)} imágenes originales y {len(imagenes_binarias)} imágenes binarias.")
    
    return imagenes_originales, imagenes_binarias

def dividir_dataset(img_base, img_mask, test_ratio=0.15, val_ratio=0.15):
    train_ratio = 1 - test_ratio - val_ratio
    pares = []
    for base in img_base:
        name = base.split('.')[0]
        mask = f"{name}_expert.png"
        
        if mask in img_mask:
            pares.append((base, mask))
            
    print(f"Total de pares encontrados: {len(pares)}")
    
    pares_arr = np.array(pares)
    seed = 42
    np.random.seed(seed)
    np.random.shuffle(pares_arr)
    n = len(pares_arr)
    
    # Índices para dividir
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_test - n_val
    
    test_set = pares_arr[:n_test]
    val_set = pares_arr[n_test:n_test + n_val]
    
    train_set = pares_arr[n_test + n_val:]
    
    dataset = {
        'train': {
            'images': train_set[:, 0].tolist(),
            'masks': train_set[:, 1].tolist(),
            'count': len(train_set)
        },
        'val': {
            'images': val_set[:, 0].tolist(),
            'masks': val_set[:, 1].tolist(),
            'count': len(val_set)
        },
        'test': {
            'images': test_set[:, 0].tolist(),
            'masks': test_set[:, 1].tolist(),
            'count': len(test_set)
        },
        'seed': seed,
        'total': n
    }
    
    print(f"\n=== DIVISIÓN DEL DATASET ===")
    print(f"Semilla utilizada: {seed}")
    print(f"Entrenamiento: {n_train} imágenes ({train_ratio*100:.1f}%)")
    print(f"Validación:    {n_val} imágenes ({val_ratio*100:.1f}%)")
    print(f"Prueba:        {n_test} imágenes ({test_ratio*100:.1f}%)")
    print(f"Total:         {n} imágenes")
    
    return dataset

#Funcion para generar histograma en escala de grises
def histo(imagenes_originales, imagenes_binarias, ruta):
    # Loop over all original images
    for img_name in imagenes_originales:
        img_path = Path(ruta) / img_name
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)  # Load as grayscale
        
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.title(f"Histograma de {img_name}")
        plt.hist(img.ravel(), bins=256, color='gray', alpha=0.7)
        plt.xlabel('Intensidad de píxel')
        plt.ylabel('Frecuencia')
        
        # For the binary mask
        mask_name = img_name.split('.')[0] + '_expert.png'
        mask_path = Path(ruta) / mask_name
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            plt.subplot(1, 2, 2)
            plt.title(f"Histograma de {mask_name}")
            plt.hist(mask.ravel(), bins=[0,128,255], color='black', alpha=0.7, rwidth=0.8)
            plt.xlabel('Intensidad de píxel')
            plt.ylabel('Frecuencia')
        
        plt.tight_layout()
        plt.show()

#Funcion para el analisis exploratiorio en los canales R, G, B de las imagenes y se analizan con la mascara respectiva.
def analizar_rgb(img_list, ruta):
    for imagen in img_list:
        img_name = imagen
        img_path = Path(ruta)/img_name
        mask_name = img_name.split('.')[0] + '_expert.png'
        mask_path = Path(ruta)/mask_name

        # if not mask_path.exists():
        #     print(f"No se encontró la máscara para {img_name}")
        #     return

        # Cargar en color (BGR en OpenCV) y convertir a RGB
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Cargar máscara en escala de grises
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # Extraer píxeles lesion y no-lesion
        lesion_pixels = img[mask > 0] # si es mayor a 0, se considera lesion
        non_lesion_pixels = img[mask == 0] # si es 0, se considera no lesion

        canales = ['R', 'G', 'B']
        colores = ['red', 'green', 'blue']

        plt.figure(figsize=(12, 6))

        for i, canal in enumerate(canales):
            plt.subplot(2, 3, i+1)
            plt.hist(lesion_pixels[:, i], bins=256, color=colores[i], alpha=0.7)
            plt.title(f"Lesión - {canal}")

            plt.subplot(2, 3, i+4)
            plt.hist(non_lesion_pixels[:, i], bins=256, color=colores[i], alpha=0.7)
            plt.title(f"No lesión - {canal}")

        plt.suptitle(f"Histogramas RGB para {img_name}")
        plt.tight_layout()
        plt.show()

        # Calcular estadisticos
        print(f"\n=== Estadísticos de {img_name} ===")
        for i, canal in enumerate(canales):
            mean_lesion = np.mean(lesion_pixels[:, i])
            std_lesion = np.std(lesion_pixels[:, i])
            mean_non = np.mean(non_lesion_pixels[:, i])
            std_non = np.std(non_lesion_pixels[:, i])

            print(f"Canal {canal}:")
            print(f"  Lesión     -> Media: {mean_lesion:.2f}, Std: {std_lesion:.2f}")
            print(f"  No-lesión  -> Media: {mean_non:.2f}, Std: {std_non:.2f}")

def analizar_rgb2(img_list, ruta):
    canales = ['R', 'G', 'B']
    colores = ['red', 'green', 'blue']

    # Diccionarios para acumular píxeles de todas las imágenes
    datos_lesion = {c: [] for c in canales}
    datos_no = {c: [] for c in canales}

    for img_name in img_list:
        img_path = Path(ruta) / img_name
        mask_name = img_name.split('.')[0] + '_expert.png'
        mask_path = Path(ruta) / mask_name

        if not mask_path.exists():
            print(f"No se encontró la máscara para {img_name}, se omite.")
            continue

        # Cargar en color (BGR -> RGB)
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Cargar máscara en escala de grises
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # Extraer píxeles
        lesion_pixels = img[mask > 0]
        non_lesion_pixels = img[mask == 0]

        # Guardar en acumuladores
        for i, canal in enumerate(canales):
            datos_lesion[canal].extend(lesion_pixels[:, i])
            datos_no[canal].extend(non_lesion_pixels[:, i])

    # === Graficar histogramas acumulados ===
    plt.figure(figsize=(12, 6))

    for i, canal in enumerate(canales):
        plt.subplot(1, 3, i+1)
        plt.hist(datos_lesion[canal], bins=256, color=colores[i], alpha=0.6, label="Lesión", density=True)
        plt.hist(datos_no[canal], bins=256, color=colores[i], alpha=0.3, label="No lesión", density=True)
        plt.title(f"Canal {canal}")
        plt.xlabel("Intensidad")
        plt.ylabel("Frecuencia")
        plt.legend()

    plt.suptitle("Histogramas RGB acumulados de todas las imágenes")
    plt.tight_layout()
    plt.show()

def estad(imagenes_originales, ruta):

    for img_name in imagenes_originales:
        img_path = Path(ruta) / img_name
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)  # Load as grayscale
        
        # estadigrafos de cada imagen en escala de grises
        mean_val = np.mean(img)
        std_val = np.std(img)
        min_val = np.min(img)
        max_val = np.max(img)
        
        print(f"Estadísticas de {img_name}:")
        print(f" - Media: {mean_val:.2f}")
        print(f" - Desviación estándar: {std_val:.2f}")
        print(f" - Mínimo: {min_val}")
        print(f" - Máximo: {max_val}")
        
        mask_name = img_name.split('.')[0] + '_expert.png'
        mask_path = Path(ruta) / mask_name
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
            mean_mask = np.mean(mask)
            std_mask = np.std(mask)
            min_mask = np.min(mask)
            max_mask = np.max(mask)
            
            print(f"Estadísticas de {mask_name}:")
            print(f" - Media: {mean_mask:.2f}")
            print(f" - Desviación estándar: {std_mask:.2f}")
            print(f" - Mínimo: {min_mask}")
            print(f" - Máximo: {max_mask}")

def boxplot(imagenes_originales, ruta):
    
    original_intensities = []
    mask_intensities = []

    # iterando con las imagenes normales y sus respectivas mascaras
    for img_name in imagenes_originales:
        img_path = Path(ruta) / img_name
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE).astype(np.float32) 

        original_intensities.append(img.ravel())

        # obtener la mascara
        mask_name = img_name.split('.')[0] + '_expert.png'
        mask_path = Path(ruta) / mask_name
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask_intensities.append(mask.ravel())
    
    data = [np.concatenate(original_intensities), np.concatenate(mask_intensities)]

    plt.figure(figsize=(8, 6))
    plt.boxplot(data, vert=True, patch_artist=True, labels=["Original", "Binary Mask"], showfliers=False)
    plt.title("Boxplot of Pixel Intensities (Original vs Binary Mask)")
    plt.ylabel("Pixel Intensity")
    plt.xlabel("Image Type")
    plt.show()

import numpy as np
import cv2
from pathlib import Path
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, roc_auc_score,
    precision_score, recall_score, f1_score, accuracy_score
)
import matplotlib.pyplot as plt

def clas_bayes(imgs_train, masks_train, imgs_test, masks_test, ruta, plot_roc=True):
    """
    Clasificador Bayesiano ingenuo (Naive Bayes) para segmentación
    basado en intensidades RGB de cada píxel.
    
    Parameters:
    -----------
    imgs_train : list
        Lista de nombres de imágenes de entrenamiento
    masks_train : list  
        Lista de nombres de máscaras de entrenamiento
    imgs_test : list
        Lista de nombres de imágenes de test
    masks_test : list
        Lista de nombres de máscaras de test  
    ruta : str
        Ruta base donde se encuentran las imágenes
    plot_roc : bool
        Si graficar la curva ROC
    
    Returns:
    --------
    dict: Diccionario con modelo, métricas y resultados
    """
    
    # ====== Construir X_train e y_train ======
    print("Cargando datos de entrenamiento...")
    X_train, y_train = [], []
    
    for img_name, mask_name in zip(imgs_train, masks_train):
        img_path = Path(ruta) / img_name
        mask_path = Path(ruta) / mask_name

        # Cargar y normalizar imagen
        img = cv2.imread(str(img_path))[:, :, ::-1].astype(np.float32) / 255.0
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8)

        X_train.append(img.reshape(-1, 3))     # (N, 3) → cada píxel = fila
        y_train.append(mask.flatten())         # (N,)  → etiqueta por píxel

    X_train = np.vstack(X_train)
    y_train = np.hstack(y_train)
    
    print(f"Datos de entrenamiento: {X_train.shape[0]} píxeles")
    print(f"Distribución de clases - No-lesión: {np.sum(y_train == 0)}, Lesión: {np.sum(y_train == 1)}")

    # ====== Entrenar clasificador ======
    print("Entrenando clasificador Bayesiano...")
    model = GaussianNB()
    model.fit(X_train, y_train)

    # ====== Evaluación en test ======
    print("Evaluando en conjunto de test...")
    X_test, y_test = [], []
    img_shapes = []  # Para reconstruir imágenes individuales
    
    for img_name, mask_name in zip(imgs_test, masks_test):
        img_path = Path(ruta) / img_name
        mask_path = Path(ruta) / mask_name

        img = cv2.imread(str(img_path))[:, :, ::-1].astype(np.float32) / 255.0
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8)
        
        img_shapes.append(img.shape[:2])  # Guardar dimensiones originales
        X_test.append(img.reshape(-1, 3))
        y_test.append(mask.flatten())

    X_test = np.vstack(X_test)
    y_test = np.hstack(y_test)

    # Predicciones binarias y probabilidades
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # Probabilidad de clase lesión

    # ====== Cálculo de métricas completas ======
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Métricas básicas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    sensitivity = recall_score(y_test, y_pred, zero_division=0)  # También llamado recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Curva ROC y AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    
    # Índice de Youden para umbral óptimo
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_idx]
    
    # ====== Cálculo de Jaccard por imagen ======
    jaccard_scores = []
    start_idx = 0
    
    for i, shape in enumerate(img_shapes):
        n_pixels = shape[0] * shape[1]
        end_idx = start_idx + n_pixels
        
        y_true_img = y_test[start_idx:end_idx]
        y_pred_img = y_pred[start_idx:end_idx]
        
        # Jaccard Index = Intersection / Union
        intersection = np.logical_and(y_true_img, y_pred_img).sum()
        union = np.logical_or(y_true_img, y_pred_img).sum()
        jaccard = intersection / union if union > 0 else 0
        jaccard_scores.append(jaccard)
        
        start_idx = end_idx
    
    jaccard_mean = np.mean(jaccard_scores)
    jaccard_std = np.std(jaccard_scores)
    
    # ====== Reportar resultados ======
    print("\n" + "="*50)
    print("EVALUACIÓN CLASIFICADOR BAYESIANO RGB")
    print("="*50)
    
    print(f"\n MATRIZ DE CONFUSIÓN:")
    print("                Predicción")
    print("              No-Les  Lesión")
    print(f"Real No-Les    {tn:6d}  {fp:6d}")
    print(f"Real Lesión    {fn:6d}  {tp:6d}")
    
    print(f"\n MÉTRICAS A NIVEL DE PÍXEL:")
    print(f"Exactitud (Accuracy):     {accuracy:.4f}")
    print(f"Precisión (Precision):    {precision:.4f}")
    print(f"Sensibilidad (Recall):    {sensitivity:.4f}")
    print(f"Especificidad:            {specificity:.4f}")
    print(f"F1-Score:                 {f1:.4f}")
    print(f"AUC-ROC:                  {auc:.4f}")
    
    print(f"\n PUNTO DE OPERACIÓN ÓPTIMO (Youden):")
    print(f"Umbral óptimo:            {optimal_threshold:.4f}")
    print(f"TPR en punto óptimo:      {tpr[optimal_idx]:.4f}")
    print(f"FPR en punto óptimo:      {fpr[optimal_idx]:.4f}")
    print(f"Índice de Youden:         {youden_index[optimal_idx]:.4f}")
    
    print(f"\n  MÉTRICAS A NIVEL DE IMAGEN:")
    print(f"Jaccard promedio:         {jaccard_mean:.4f} ± {jaccard_std:.4f}")
    print(f"Jaccard mínimo:           {min(jaccard_scores):.4f}")
    print(f"Jaccard máximo:           {max(jaccard_scores):.4f}")
    
    print(f"\n REPORTE DETALLADO POR CLASE:")
    print(classification_report(y_test, y_pred, target_names=['No-lesión', 'Lesión']))
    
    # ====== Gráfico de curva ROC ======
    if plot_roc:
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'Curva ROC (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Clasificador aleatorio')
        
        # Marcar punto óptimo
        plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=8,
                label=f'Punto óptimo (Youden = {youden_index[optimal_idx]:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Tasa de Falsos Positivos (FPR)', fontsize=12)
        plt.ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=12)
        plt.title('Curva ROC - Clasificador Bayesiano RGB', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    # ====== Preparar resultados para retorno ======
    resultados = {
        'modelo': model,
        'metricas': {
            'accuracy': accuracy,
            'precision': precision,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'f1_score': f1,
            'auc': auc,
            'jaccard_mean': jaccard_mean,
            'jaccard_std': jaccard_std,
            'jaccard_scores': jaccard_scores
        },
        'curva_roc': {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'auc': auc
        },
        'punto_optimo': {
            'threshold': optimal_threshold,
            'tpr': tpr[optimal_idx],
            'fpr': fpr[optimal_idx],
            'youden': youden_index[optimal_idx]
        },
        'predicciones': {
            'y_pred': y_pred,
            'y_proba': y_proba,
            'y_test': y_test
        },
        'matriz_confusion': cm,
        'img_shapes': img_shapes
    }
    
    print(f"\n✅ Clasificación completada. Modelo y métricas guardados en el diccionario de retorno.")
    
    return resultados

def mostrar_ejemplo_segmentacion(resultados, imgs_test, ruta, idx_imagen=0):
    """
    Muestra un ejemplo de segmentación para una imagen específica
    
    Parametros
    -----------
    resultados : dict
        Diccionario retornado por clas_bayes()
    imgs_test : list
        Lista de nombres de imágenes de test
    ruta : str
        Ruta base de las imágenes
    idx_imagen : int
        Índice de la imagen a mostrar
    """
    
    # Cargar imagen original
    img_path = Path(ruta) / imgs_test[idx_imagen]
    img_original = cv2.imread(str(img_path))[:, :, ::-1]
    
    # Obtener predicción para esta imagen
    img_shapes = resultados['img_shapes']
    y_pred = resultados['predicciones']['y_pred']
    
    # Calcular índices para esta imagen específica
    start_idx = sum(shape[0] * shape[1] for shape in img_shapes[:idx_imagen])
    end_idx = start_idx + img_shapes[idx_imagen][0] * img_shapes[idx_imagen][1]
    
    # Reconstruir máscara predicha
    pred_mask = y_pred[start_idx:end_idx].reshape(img_shapes[idx_imagen])
    
    # Visualizar
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_original)
    axes[0].set_title('Imagen Original', fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(pred_mask, cmap='gray')
    axes[1].set_title('Segmentación Predicha', fontweight='bold')
    axes[1].axis('off')
    
    # Overlay
    overlay = img_original.copy()
    overlay[pred_mask == 1] = [255, 0, 0]  # Lesión en rojo
    axes[2].imshow(cv2.addWeighted(img_original, 0.7, overlay, 0.3, 0))
    axes[2].set_title('Overlay (Lesión en Rojo)', fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report, jaccard_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path

def clas_kmeans(imgs_test, masks_test, ruta, n_clusters=2, random_state=42, save_dir="./Resultados_KMeans"):
    """
    Clasificador NO supervisado usando K-Means sobre intensidades RGB.
    Guarda las segmentaciones en archivos PNG.
    """
    
    print("=== SEGMENTACIÓN NO SUPERVISADA CON K-MEANS ===")
    
    # Crear carpeta de resultados si no existe
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    all_metrics = []
    segmentaciones = []
    
    for img_name, mask_name in zip(imgs_test, masks_test):
        img_path = Path(ruta) / img_name
        mask_path = Path(ruta) / mask_name
        
        # cargar imagen
        img = cv2.imread(str(img_path))[:, :, ::-1].astype(np.float32) / 255.0
        h, w, _ = img.shape
        X = img.reshape(-1, 3)   # píxeles como filas
        
        # aplicar k-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X)
        seg_img = labels.reshape(h, w)
        
        # Normalizar etiquetas (0/1)
        # Asumimos que la clase "lesión" es la más pequeña en área
        lesion_label = 1 if np.sum(seg_img == 1) < np.sum(seg_img == 0) else 0
        pred_mask = (seg_img == lesion_label).astype(np.uint8) * 255  # convertir a 0/255
        
        segmentaciones.append(pred_mask)
        
        # === Guardar resultado como PNG ===
        out_name = Path(img_name).stem + "_kmeans.png"
        out_path = Path(save_dir) / out_name
        cv2.imwrite(str(out_path), pred_mask)
        
        # Evaluar si existe la máscara real
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.uint8)
            
            acc = accuracy_score(mask.flatten(), (pred_mask > 127).flatten())
            f1 = f1_score(mask.flatten(), (pred_mask > 127).flatten())
            jaccard = jaccard_score(mask.flatten(), (pred_mask > 127).flatten())
            
            all_metrics.append((acc, f1, jaccard))
    
    # Promediar métricas
    if all_metrics:
        acc_mean, f1_mean, jacc_mean = np.mean(all_metrics, axis=0)
        print(f"Exactitud promedio: {acc_mean:.4f}")
        print(f"F1-score promedio: {f1_mean:.4f}")
        print(f"Jaccard promedio: {jacc_mean:.4f}")
    else:
        print("No se encontraron máscaras para evaluar.")
    
    print(f"\n✅ Segmentaciones guardadas en: {save_dir}")
    
    return segmentaciones

def mostrar_ejemplo_kmeans(segmentaciones, imgs_test, ruta, idx_imagen=0):
    """
    Mostrar una segmentación obtenida con K-means
    """
    img_path = Path(ruta) / imgs_test[idx_imagen]
    img_original = cv2.imread(str(img_path))[:, :, ::-1]
    
    pred_mask = segmentaciones[idx_imagen]
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(img_original)
    axes[0].set_title("Imagen Original")
    axes[0].axis("off")
    
    axes[1].imshow(pred_mask, cmap="gray")
    axes[1].set_title("Segmentación K-means")
    axes[1].axis("off")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    img_base, img_mask = cargar_imagenes_dermastocopia(dataset_ruta)

    dataset = dividir_dataset(img_base, img_mask)

    # histo(img_base, img_mask, dataset_ruta)
    # analizar_rgb(img_base, dataset_ruta)
    # analizar_rgb2(img_base, dataset_ruta)
    # estad(img_base, img_mask, dataset_ruta)
    #boxplot(img_base, dataset_ruta)
    
    #bayes = clas_bayes(dataset['train']['images'], dataset['train']['masks'], dataset['test']['images'], dataset['test']['masks'], dataset_ruta)
    #mostrar_ejemplo_segmentacion(bayes, dataset['test']['images'], dataset_ruta, idx_imagen=0)

    kmeans_segs = clas_kmeans(
        dataset['test']['images'],
        dataset['test']['masks'],
        dataset_ruta
    )

    mostrar_ejemplo_kmeans(kmeans_segs, dataset['test']['images'], dataset_ruta, idx_imagen=0)