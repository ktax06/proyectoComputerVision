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
    image: str

class RegisterRequest(BaseModel):
    image: str
    name: str

class APIResponse(BaseModel):
    success: bool
    message: str
    data: Dict = None

# Base de datos simple en memoria (en producción usarías una DB real)
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
        # Convertir numpy arrays a listas para JSON
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

def get_face_encoding(image_array: np.ndarray) -> np.ndarray:
    """Obtener encoding facial de una imagen"""
    try:
        # Detectar caras en la imagen
        face_locations = face_recognition.face_locations(image_array)
        
        if len(face_locations) == 0:
            raise ValueError("No se detectó ningún rostro en la imagen")
        
        if len(face_locations) > 1:
            print("Advertencia: Se detectaron múltiples rostros, usando el primero")
        
        # Obtener encodings faciales
        face_encodings = face_recognition.face_encodings(image_array, face_locations)
        
        if len(face_encodings) == 0:
            raise ValueError("No se pudo extraer características faciales")
        
        return face_encodings[0]
    
    except Exception as e:
        raise ValueError(f"Error en reconocimiento facial: {str(e)}")

def find_matching_user(face_encoding: np.ndarray, tolerance: float = 0.6) -> str:
    """Buscar usuario que coincida con el encoding facial"""
    for user_id, user_data in users_db.items():
        stored_encoding = user_data['face_encoding']
        
        # Comparar encodings
        matches = face_recognition.compare_faces([stored_encoding], face_encoding, tolerance=tolerance)
        
        if matches[0]:
            # Calcular distancia para mayor precisión
            distance = face_recognition.face_distance([stored_encoding], face_encoding)[0]
            print(f"Match encontrado para {user_data['name']} con distancia: {distance:.3f}")
            return user_id
    
    return None

# Cargar usuarios al iniciar
load_users()

@app.get("/")
async def root():
    return {"message": "Facial Recognition API está funcionando", "users_count": len(users_db)}

@app.post("/register", response_model=APIResponse)
async def register_face(request: RegisterRequest):
    """Registrar un nuevo rostro"""
    try:
        # Validar entrada
        if not request.name.strip():
            raise ValueError("El nombre es requerido")
        
        if not request.image:
            raise ValueError("La imagen es requerida")
        
        # Procesar imagen
        image_array = base64_to_image(request.image)
        
        # Obtener encoding facial
        face_encoding = get_face_encoding(image_array)
        
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
        
        # Persistir datos
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
    """Autenticar usuario con reconocimiento facial"""
    try:
        if not request.image:
            raise ValueError("La imagen es requerida")
        
        if len(users_db) == 0:
            return APIResponse(
                success=False,
                message="No hay usuarios registrados. Registra tu rostro primero."
            )
        
        # Procesar imagen
        image_array = base64_to_image(request.image)
        
        # Obtener encoding facial
        face_encoding = get_face_encoding(image_array)
        
        # Buscar usuario coincidente
        user_id = find_matching_user(face_encoding)
        
        if user_id:
            user_data = users_db[user_id]
            print(f"Login exitoso para: {user_data['name']}")
            
            return APIResponse(
                success=True,
                message=f"¡Bienvenido, {user_data['name']}!",
                data={
                    "user_id": user_id,
                    "name": user_data['name'],
                    "login_time": str(np.datetime64('now'))
                }
            )
        else:
            return APIResponse(
                success=False,
                message="Rostro no reconocido. Verifica tu posición o registra tu rostro."
            )
    
    except ValueError as e:
        return APIResponse(success=False, message=str(e))
    except Exception as e:
        print(f"Error en login: {e}")
        return APIResponse(success=False, message="Error interno del servidor")

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

if __name__ == "__main__":
    import uvicorn
    print("Iniciando servidor de reconocimiento facial...")
    print("Usuarios registrados:", len(users_db))
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)


class UpdateUserRequest(BaseModel):
    name: str

@app.put("/users/{user_id}", response_model=APIResponse)
async def update_user(user_id: str, request: UpdateUserRequest):
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
    try:
        image_array = base64_to_image(request.image)
        face_encoding = get_face_encoding(image_array)
        user_id = find_matching_user(face_encoding)
        
        if user_id:
            user = users_db[user_id]
            return APIResponse(
                success=True,
                message=f"Rostro reconocido: {user['name']}",
                data={"user_id": user_id, "name": user['name']}
            )
        else:
            return APIResponse(success=False, message="Rostro no encontrado")
    except Exception as e:
        return APIResponse(success=False, message=str(e))
