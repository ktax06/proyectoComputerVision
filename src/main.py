import cv2

# Ruta de la imagen
ruta_imagen = './data/Arnold-Schwarzenegger-Beach-Pose-Poster.jpg'  # Reemplaza con la ruta correcta

# Cargar la imagen
imagen = cv2.imread(ruta_imagen)

# Verificar si la imagen se cargó correctamente
if imagen is None:
    print("Error al cargar la imagen. Verifica la ruta.")
else:
    # Mostrar la imagen en una ventana
    cv2.imshow('Imagen', imagen)

    # Esperar hasta que se presione una tecla
    cv2.waitKey(0)

    # Cerrar todas las catalinda
    cv2.destroyAllWindows()
