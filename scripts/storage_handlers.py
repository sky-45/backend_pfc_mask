from azure.storage.blob import BlobClient, BlobServiceClient

az_conection_str = "DefaultEndpointsProtocol=https;AccountName=photosapppfc;AccountKey=OwPugyOmFGhhqCyHNNmE3K40JYIzliaNfvIVgxdcUdlk6VUf28sl+PcPNH1Wb1OSQKptBsgAHs/PEYp6ETEvGA==;EndpointSuffix=core.windows.net"
blob_service_client = BlobServiceClient.from_connection_string(conn_str = az_conection_str)
try:
    container_client = blob_service_client.get_container_client(container='photos')
    container_client.get_container_properties()

except Exception as e:
    print("cant connect to container")

def upload_photos(id):
    path_original_photo = ("test.jpg","org_photo")
    path_fnal_photo = ("inference/output/test.jpg","pred_photo")
    filenames = []
    for path_photo in (path_original_photo,path_fnal_photo):
        file_name = id + "-"+path_photo[1] + ".jpg"
        with open(path_photo[0], "rb") as img:
            container_client.upload_blob(file_name,img)
        filenames.append("https://photosapppfc.blob.core.windows.net/photos/"+file_name)
        
    return filenames # [original, modified]
