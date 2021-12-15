# Imprtando modules
from typing import Optional
from fastapi import Depends, FastAPI, HTTPException, status, File, UploadFile
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse,Response

# Importando scripts
from scripts import reports
import model
from datetime import datetime

fake_users_db = {
    "johndoe": {
        "username": "johndoe",
        "full_name": "John Doe",
        "email": "johndoe@example.com",
        "hashed_password": "fakehashedsecret",
        "disabled": False,
    },
    "alice": {
        "username": "alice",
        "full_name": "Alice Wonderson",
        "email": "alice@example.com",
        "hashed_password": "fakehashedsecret2",
        "disabled": True,
    },
    "admin": {
        "username": "admin",
        "full_name": "Name lastname",
        "email": "admin@admin.com",
        "hashed_password": "fakehashedadmin",
        "disabled": False,
    },
}

app = FastAPI()

origins = ['*']
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def fake_hash_password(password: str):
    return "fakehashed" + password


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

### DEFINIMOS MODELOS
class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None


class UserInDB(User):
    hashed_password: str
class NewReport(BaseModel):
    date: str
    url: str
    description: str




def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)


def fake_decode_token(token):
    # This doesn't provide any security at all
    # Check the next version
    user = get_user(fake_users_db, token)
    return user



async def get_current_user(token: str = Depends(oauth2_scheme)):
    user = fake_decode_token(token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user
async def get_current_active_user(current_user: User = Depends(get_current_user)):

    if current_user.disabled:

        raise HTTPException(status_code=400, detail="Inactive user")

    return current_user


############################################# DEFINIMOS ENDPOINTS ########################################
@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user_dict = fake_users_db.get(form_data.username)
    if not user_dict:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    user = UserInDB(**user_dict)
    hashed_password = fake_hash_password(form_data.password)
    if not hashed_password == user.hashed_password:
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    return {"access_token": user.username, "token_type": "bearer"}

@app.get("/users/me")
async def read_users_me(current_user: User = Depends(get_current_active_user)):

    return current_user

########################### fetch and modify data

@app.get("/data")
async def get_all_data():
    data = reports.get_logs()
    return data

@app.get("/data/{date}")
async def get_data_since(date:str,current_user: User= Depends(get_current_active_user)):
    #reemplazamos la fecha dd-mm-yyyy a dd/mm/yyyy, ej: 16-09-2019 a 16/09/2019
    data = reports.get_reports(date.replace("-","/"))
    return data


########################### upload data
@app.post("/upload_ret_image/")
async def create_up_ret_file(file: UploadFile = File(...)):
    contents = await file.read()  # <-- Important!
    responsexd,files_cloud = model.get_predictions_img(contents)
    data_log =  {"id_log":reports.get_id(),
                "date_log":datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
                "url_org_img":files_cloud[0],
                "url_pred_img":files_cloud[1],
                "detections_data":responsexd
                }
    reports.add_reports(data_log)
    return {"status":"succesfull"}
