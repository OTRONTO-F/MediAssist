# HealthCare ChatBot

HealthCare ChatBot เป็นโปรแกรม Python ที่ใช้สร้างและใช้งานแชทบอทด้านการรักษาสุขภาพ โปรแกรมนี้สามารถวิเคราะห์อาการทางการแพทย์และให้ข้อมูลการวินิจฉัยโรคโดยใช้ Decision Tree และ SVM (Support Vector Machine) และให้คำแนะนำเกี่ยวกับโรคที่เป็นไปได้ตามอาการที่ผู้ใช้รายงาน.

## วิธีใช้

1. ติดตั้งไลบรารีและโมดูลที่จำเป็นโดยใช้คำสั่งต่อไปนี้:


2. โหลดข้อมูลการอบรมและการทดสอบจากไฟล์ CSV โดยใช้คำสั่งต่อไปนี้:

```python
training = pd.read_csv('Data/Training.csv')
testing = pd.read_csv('Data/Testing.csv')

