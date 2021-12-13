from datetime import datetime, date

report_list = [
        {"date":"11/09/2019","id":1,"url":"www.google.com","description":"Imagen persona 1"},
        {"date":"11/09/2019","id":2,"url":"www.google.com","description":"Imagen persona 2"},
        {"date":"12/09/2019","id":3,"url":"www.google.com","description":"Imagen persona 3"},
        {"date":"12/09/2019","id":4,"url":"www.google.com","description":"Imagen persona 4"},
        {"date":"13/09/2019","id":5,"url":"www.google.com","description":"Imagen persona 5"},
        {"date":"13/09/2019","id":6,"url":"www.google.com","description":"Imagen persona 6"},
        {"date":"14/09/2019","id":7,"url":"www.google.com","description":"Imagen persona 7"},
        {"date":"14/09/2019","id":8,"url":"www.google.com","description":"Imagen persona 8"},
        {"date":"15/09/2019","id":9,"url":"www.google.com","description":"Imagen persona 9"},
        {"date":"15/09/2019","id":10,"url":"www.google.com","description":"Imagen persona 10"},
        {"date":"16/09/2019","id":11,"url":"www.google.com","description":"Imagen persona 11"},
        {"date":"16/09/2019","id":12,"url":"www.google.com","description":"Imagen persona 12"},
        {"date":"16/09/2019","id":13,"url":"www.google.com","description":"Imagen persona 13"},
    ]
def get_reports(date:str=datetime.now().strftime("%d/%m/%Y")):
    data = [register for register in report_list if datetime.strptime(register["date"], "%d/%m/%Y") >= datetime.strptime(date, "%d/%m/%Y")]
    return {"data":data}

def add_reports(report):
    new_id = len(report_list)+1
    report_list.append({"date":report.date,"id":new_id,"url":report.url,"description":report.description})
    return report_list[new_id-1]
