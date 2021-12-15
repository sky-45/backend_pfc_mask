from datetime import datetime, date

report_logs = [
        {"id_log":1,
        "date_log":"25-06-2021 07:58:56",
        "url_org_img":"https://photosapppfc.blob.core.windows.net/photos/14-org_photo.jpg",
        "url_pred_img":"https://photosapppfc.blob.core.windows.net/photos/14-pred_photo.jpg",
        "detections_data":[
            {
                "id_person":1,
                "box_xy":[1,2,3,4],
                "yolo_acc":0.89,
                "mask_acc":0.60
            },
            {
                "id_person":2,
                "box_xy":[1,2,3,4],
                "yolo_acc":0.89,
                "mask_acc":0.60
            }
            ]
        },
        {"id_log":2,
        "date_log":"25-07-2021 07:58:56",
        "url_org_img":"https://photosapppfc.blob.core.windows.net/photos/14-org_photo.jpg",
        "url_pred_img":"https://photosapppfc.blob.core.windows.net/photos/14-pred_photo.jpg",
        "detections_data":[
            {
                "id_person":1,
                "box_xy":[1,2,3,4],
                "yolo_acc":0.89,
                "mask_acc":0.60
            },
            {
                "id_person":2,
                "box_xy":[1,2,3,4],
                "yolo_acc":0.89,
                "mask_acc":0.60
            }
            ]
        },
        
    ]
def get_reports(date:str=datetime.now().strftime("%d/%m/%Y")):
    data = [register for register in report_logs if datetime.strptime(register["date"], "%d/%m/%Y") >= datetime.strptime(date, "%d/%m/%Y")]
    return {"data":data}
def get_logs():
    return report_logs

def add_reports(report):
    report_logs.append(report)
def get_id():
    return str(len(report_logs)+1)

