from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import face_recognition
import numpy as np
import base64
import io
from PIL import Image
import json
import os
from typing import List, Dict

# Creación de la aplicación FastAPI
app = FastAPI(title="Facial Recognition API", version="1.0.0")

# Configurar CORS para permitir requests desde el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4321", "http://localhost:3000"],  # Astro dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelos de datos
class LoginRequest(BaseModel):
    image: str  # Imagen en formato base64

class RegisterRequest(BaseModel):
    image: str  # Imagen en formato base64
    name: str   # Nombre del usuario a registrar

class APIResponse(BaseModel):
    success: bool      # Indica si la operación tuvo éxito
    message: str       # Mensaje descriptivo
    data: Dict = None  # Datos adicionales (opcional)

# Base de datos temporal en memoria
users_db = {}
USERS_FILE = "users_data.json"

def load_users():
    """Cargar usuarios desde archivo JSON"""
    global users_db
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, 'r') as f:
                data = json.load(f)
                # Convertir encodings de lista a numpy array
                for user_id, user_data in data.items():
                    if 'face_encoding' in user_data:
                        user_data['face_encoding'] = np.array(user_data['face_encoding'])
                users_db = data
            print(f"Cargados {len(users_db)} usuarios desde archivo")
        except Exception as e:
            print(f"Error cargando usuarios: {e}")
            users_db = {}
    else:
        users_db = {}

def save_users():
    """Guardar usuarios en archivo JSON"""
    try:
        # Convertir numpy arrays a listas para poder guardarlos en JSON
        data_to_save = {}
        for user_id, user_data in users_db.items():
            data_copy = user_data.copy()
            if 'face_encoding' in data_copy:
                data_copy['face_encoding'] = data_copy['face_encoding'].tolist()
            data_to_save[user_id] = data_copy
        
        with open(USERS_FILE, 'w') as f:
            json.dump(data_to_save, f, indent=2)
        print("Usuarios guardados en archivo")
    except Exception as e:
        print(f"Error guardando usuarios: {e}")

def base64_to_image(base64_string: str) -> np.ndarray:
    """Convertir imagen base64 a array numpy"""
    try:
        # Remover el prefijo data:image/jpeg;base64, si existe
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
        
        # Decodificar base64
        image_data = base64.b64decode(base64_string)
        
        # Convertir a PIL Image
        image = Image.open(io.BytesIO(image_data))
        
        # Convertir a RGB si es necesario
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convertir a numpy array
        return np.array(image)
    
    except Exception as e:
        raise ValueError(f"Error procesando imagen: {str(e)}")

def get_face_encoding(image_array: np.ndarray, allow_multiple: bool = False) -> tuple:
    """
    Extrae características faciales de una imagen
    Returns: (face_encoding, face_count) si allow_multiple=False
             (face_encodings_list, face_count) si allow_multiple=True
    """
    try:
        # Detectar caras en la imagen
        face_locations = face_recognition.face_locations(image_array)
        face_count = len(face_locations)
        
        # Validar que se encontró al menos un rostro
        if face_count == 0:
            raise ValueError("No se detectó ningún rostro en la imagen")
        
        # Para registro: solo permitir UN rostro
        if not allow_multiple and face_count > 1:
            raise ValueError(f"Se detectaron {face_count} rostros. Para registrarte, asegúrate de que solo tu rostro esté visible en la cámara.")
        
        # Obtener encodings faciales
        face_encodings = face_recognition.face_encodings(image_array, face_locations)
        
        if len(face_encodings) == 0:
            raise ValueError("No se pudo extraer características faciales")
        
        if allow_multiple:
            return face_encodings, face_count
        else:
            return face_encodings[0], face_count
    
    except Exception as e:
        raise ValueError(f"Error en reconocimiento facial: {str(e)}")

def find_matching_user(face_encoding: np.ndarray, tolerance: float = 0.6) -> str:
    """Buscar usuario que coincida con el encoding facial"""
    for user_id, user_data in users_db.items():
        stored_encoding = user_data['face_encoding']
        
        # Comparar encodings
        matches = face_recognition.compare_faces([stored_encoding], face_encoding, tolerance=tolerance)
        
        if matches[0]:
            # Actualizar datos de login
            user_data['last_login'] = str(np.datetime64('now'))
            user_data['login_count'] = user_data.get('login_count', 0) + 1
            # Guardar cambios
            save_users()
            # Calcular distancia para mayor precisión
            distance = face_recognition.face_distance([stored_encoding], face_encoding)[0]
            print(f"Match encontrado para {user_data['name']} con distancia: {distance:.3f}")
            return user_id
    # Si no hay coincidencias
    return None

# Cargar usuarios al iniciar
load_users()

# --- RUTAS DE LA API ---
@app.get("/")
async def root():
    """Ruta principal para verificar que el servidor está funcionando"""
    return {"message": "Facial Recognition API está funcionando", "users_count": len(users_db)}

@app.post("/register", response_model=APIResponse)
async def register_face(request: RegisterRequest):
    """Registrar un nuevo rostro - SOLO permite UN rostro"""
    try:
        # Validar entrada
        if not request.name.strip():
            raise ValueError("El nombre es requerido")
        
        if not request.image:
            raise ValueError("La imagen es requerida")
        
        # Procesar imagen
        image_array = base64_to_image(request.image)
        
        # Obtener encoding facial - NO permitir múltiples rostros
        face_encoding, face_count = get_face_encoding(image_array, allow_multiple=False)
        
        # Verificar si el rostro ya está registrado
        existing_user = find_matching_user(face_encoding, tolerance=0.5)
        if existing_user:
            existing_name = users_db[existing_user]['name']
            return APIResponse(
                success=False,
                message=f"Este rostro ya está registrado como: {existing_name}"
            )
        
        # Generar ID único
        user_id = f"user_{len(users_db) + 1}_{hash(request.name) % 10000}"
        
        # Guardar usuario
        users_db[user_id] = {
            'name': request.name.strip(),
            'face_encoding': face_encoding,
            'registered_at': str(np.datetime64('now'))
        }
        
        # Guardar cambios
        save_users()
        
        print(f"Usuario registrado: {request.name} (ID: {user_id})")
        
        return APIResponse(
            success=True,
            message=f"Rostro registrado exitosamente para {request.name}",
            data={"user_id": user_id, "name": request.name}
        )
    
    except ValueError as e:
        return APIResponse(success=False, message=str(e))
    except Exception as e:
        print(f"Error en registro: {e}")
        return APIResponse(success=False, message="Error interno del servidor")

@app.post("/login", response_model=APIResponse)
async def login_with_face(request: LoginRequest):
    """Autenticar usuario con reconocimiento facial - permite múltiples rostros"""
    try:
        # Validaciones básicas
        if not request.image:
            raise ValueError("La imagen es requerida")
        
        if len(users_db) == 0:
            return APIResponse(
                success=False,
                message="No hay usuarios registrados. Registra tu rostro primero."
            )
        
        # Procesar imagen
        image_array = base64_to_image(request.image)
        
        # Obtener encodings faciales - permitir múltiples rostros para login
        face_encodings, face_count = get_face_encoding(image_array, allow_multiple=True)
        
        # Buscar coincidencias con todos los rostros detectados
        for i, face_encoding in enumerate(face_encodings):
            user_id = find_matching_user(face_encoding)
            if user_id:
                user_data = users_db[user_id]
                message = f"¡Bienvenido, {user_data['name']}!"
                if face_count > 1:
                    message += f" (Se detectaron {face_count} rostros en total)"
                
                print(f"Login exitoso para: {user_data['name']} - Rostro {i+1} de {face_count}")
                
                return APIResponse(
                    success=True,
                    message=message,
                    data={
                        "user_id": user_id,
                        "name": user_data['name'],
                        "login_time": str(np.datetime64('now')),
                        "faces_detected": face_count
                    }
                )
        
        # Si no se encontraron coincidencias
        message = "Rostro no reconocido. Verifica tu posición o registra tu rostro."
        if face_count > 1:
            message += f" ({face_count} rostros detectados)"
        
        return APIResponse(success=False, message=message)
    
    except ValueError as e:
        return APIResponse(success=False, message=str(e))
    except Exception as e:
        print(f"Error en login: {e}")
        return APIResponse(success=False, message="Error interno del servidor")

# Nueva ruta para verificar cuántos rostros hay en una imagen
@app.post("/detect-faces", response_model=APIResponse)
async def detect_faces(request: LoginRequest):
    """Detectar número de rostros en una imagen"""
    try:
        if not request.image:
            raise ValueError("La imagen es requerida")
        
        # Procesar imagen
        image_array = base64_to_image(request.image)
        
        # Detectar rostros
        face_locations = face_recognition.face_locations(image_array)
        face_count = len(face_locations)
        
        if face_count == 0:
            return APIResponse(
                success=False,
                message="No se detectó ningún rostro",
                data={"face_count": 0}
            )
        elif face_count == 1:
            return APIResponse(
                success=True,
                message="Un rostro detectado - listo para registro",
                data={"face_count": 1}
            )
        else:
            return APIResponse(
                success=False,
                message=f"Se detectaron {face_count} rostros - para registrarte debe haber solo uno",
                data={"face_count": face_count}
            )
    
    except Exception as e:
        return APIResponse(success=False, message=str(e))

@app.get("/users")
async def list_users():
    """Listar usuarios registrados (solo nombres)"""
    users_list = []
    for user_id, user_data in users_db.items():
        users_list.append({
            "id": user_id,
            "name": user_data['name'],
            "registered_at": user_data.get('registered_at', 'Unknown')
        })
    
    return {
        "users": users_list,
        "total": len(users_list)
    }

class LogoutRequest(BaseModel):
    user_id: str  # ID del usuario que cierra sesión

@app.post("/logout", response_model=APIResponse)
async def logout_user(request: LogoutRequest):
    """Registrar cierre de sesión del usuario"""
    try:
        if request.user_id in users_db:
            # Registrar hora de cierre de sesión
            users_db[request.user_id]['last_logout'] = str(np.datetime64('now'))
            save_users()  # Guardar cambios en el archivo
            
            return APIResponse(
                success=True,
                message="Sesión cerrada exitosamente"
            )
        else:
            return APIResponse(
                success=False,
                message="Usuario no encontrado"
            )
    except Exception as e:
        print(f"Error en logout: {e}")
        return APIResponse(success=False, message="Error interno del servidor")

@app.delete("/users/{user_id}")
async def delete_user(user_id: str):
    """Eliminar usuario"""
    if user_id in users_db:
        deleted_user = users_db.pop(user_id)
        save_users()
        return APIResponse(
            success=True,
            message=f"Usuario {deleted_user['name']} eliminado correctamente"
        )
    else:
        return APIResponse(success=False, message="Usuario no encontrado")

class UpdateUserRequest(BaseModel):
    name: str  # Nuevo nombre para el usuario

@app.put("/users/{user_id}", response_model=APIResponse)
async def update_user(user_id: str, request: UpdateUserRequest):
    """Actualizar información de un usuario"""
    print("Usuarios registrados:", users_db.keys())
    if user_id in users_db:
        old_name = users_db[user_id]['name']
        users_db[user_id]['name'] = request.name.strip()
        save_users()
        return APIResponse(
            success=True,
            message=f"Nombre actualizado de '{old_name}' a '{request.name}'"
        )
    return APIResponse(success=False, message="Usuario no encontrado")

@app.get("/users/{user_id}", response_model=APIResponse)
async def get_user(user_id: str):
    """Obtener información de un usuario específico"""
    if user_id in users_db:
        user = users_db[user_id]
        return APIResponse(
            success=True,
            message="Usuario encontrado",
            data={
                "user_id": user_id,
                "name": user['name'],
                "registered_at": user.get('registered_at')
            }
        )
    return APIResponse(success=False, message="Usuario no encontrado")

@app.post("/check-face", response_model=APIResponse)
async def check_face(request: LoginRequest):
    """Verificar si un rostro está registrado sin iniciar sesión"""
    try:
        image_array = base64_to_image(request.image)
        face_encoding, face_count = get_face_encoding(image_array, allow_multiple=True)
        
        # Si hay múltiples rostros, usar el primero
        if isinstance(face_encoding, list):
            face_encoding = face_encoding[0]
        
        user_id = find_matching_user(face_encoding)
        
        if user_id:
            user = users_db[user_id]
            message = f"Rostro reconocido: {user['name']}"
            if face_count > 1:
                message += f" ({face_count} rostros detectados)"
            
            return APIResponse(
                success=True,
                message=message,
                data={"user_id": user_id, "name": user['name'], "faces_detected": face_count}
            )
        else:
            message = "Rostro no encontrado"
            if face_count > 1:
                message += f" ({face_count} rostros detectados)"
            return APIResponse(success=False, message=message)
    except Exception as e:
        return APIResponse(success=False, message=str(e))

if __name__ == "__main__":
    import uvicorn
    print("Iniciando servidor de reconocimiento facial...")
    print("Usuarios registrados:", len(users_db))
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)