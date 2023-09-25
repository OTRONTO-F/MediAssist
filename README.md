# MediAssist

MediAssist is a Python program dedicated to improving healthcare by delivering disease recommendations in response to user-reported symptoms. This powerful tool employs Decision Trees and Support Vector Machine (SVM) for accurate disease diagnosis and provides comprehensive guidance on the most suitable treatment methods, all based on the user's reported symptoms.

Key Features:

1. **Symptom-Based Diagnosis:** MediAssist uses state-of-the-art Decision Trees and SVM algorithms to accurately diagnose diseases based on the symptoms provided by users.

2. **Personalized Recommendations:** Tailored recommendations for disease management and treatment are generated, ensuring users receive the most appropriate advice for their specific symptoms.

3. **Comprehensive Information:** Users can access detailed information about various diseases, including causes, symptoms, and available treatment options.

4. **User-Friendly Interface:** The program offers an intuitive and user-friendly interface, making it accessible for individuals with varying levels of technical expertise.

5. **Privacy and Security:** MediAssist prioritizes user data privacy and employs robust security measures to safeguard sensitive health information.

6. **Regular Updates:** The program stays up-to-date with the latest medical research and guidelines to ensure accurate and relevant recommendations.

7. **Sending Diagnosis Results via Email with MediAssist:** MediAssist allows you to send diagnosis results via email to receive immediate information and recommendations in your email inbox. Here's how it works.

## วิธีใช้

1. **ติดตั้งไลบรารีและโมดูลที่จำเป็น**
    ```
    pip install -r requirements.txt
    ```

2. **โหลดข้อมูลการอบรมและการทดสอบ**
   โปรแกรมต้องการข้อมูลการอบรมและการทดสอบจากไฟล์ CSV เพื่อทำการวิเคราะห์อาการและการวินิจฉัยโรค โปรดดาวน์โหลดและนำเข้าข้อมูลดังนี้:
   - `Data/Training.csv`: ไฟล์ CSV สำหรับข้อมูลการอบรม
   - `Data/Testing.csv`: ไฟล์ CSV สำหรับข้อมูลการทดสอบ

**หมายเหตุ**: คุณสามารถเพิ่มข้อมูลเพิ่มเติมเกี่ยวกับอาการทางการแพทย์และโรคในไฟล์ CSV ตามที่คุณต้องการ.

## ตัวอย่างการใช้งาน

### วิธีวิเคราะห์อาการ

1. เริ่มต้นโปรแกรม MediAssist.
2. ป้อนชื่อของคุณเพื่อเริ่มการสนทนา.
3. ป้อนอาการที่คุณกำลังทดสอบ.
4. โปรแกรมจะทำการวิเคราะห์อาการและแสดงข้อมูลการวินิจฉัยโรคที่เป็นไปได้.

### ตัวอย่างผลลัพธ์

- การป้อน "ปวดท้อง" อาจส่งผลให้โปรแกรมแนะนำว่าคุณอาจมีโรคท้องผูก.
- การป้อน "ไข้, คัดจมูก" อาจส่งผลให้โปรแกรมแนะนำว่าคุณอาจมีโรคหวัดหรือไข้หวัด.

**หมายเหตุ**: อย่าลืมป้อนข้อมูลอาการของคุณเพื่อรับข้อมูลการวินิจฉัยที่แม่นยำ!

## การวินิจฉัยโรค

MediAssist สามารถระบุความรุนแรงของอาการและแนะนำวิธีการรักษาตามผลการวินิจฉัย. โปรแกรมสามารถทำได้ด้วยการวิเคราะห์อาการที่คุณป้อนและเปรียบเทียบกับฐานข้อมูลอาการและโรคที่มีอยู่.

## คำแนะนำการรักษา

หากโปรแกรมตรวจพบว่าคุณอาจมีโรคใดโรคหนึ่ง โปรแกรมจะแนะนำมาตรการที่ควรทำเพื่อรักษาโรคนั้น ๆ อย่างถูกต้อง.

## การส่งผลผ่านอีเมล

คุณสามารถใช้ MediAssist ส่งผลการวินิจฉัยทางอีเมล เพื่อให้คุณได้รับข้อมูลและคำแนะนำได้ทันทีผ่านทางอีเมลของคุณ.

## ข้อมูลเพิ่มเติม

- [คู่มือการใช้งาน](user_guide.md): คู่มือและข้อมูลเพิ่มเติมเกี่ยวกับการใช้งาน MediAssist.

## ข้อตกลงและเงื่อนไข

โปรดอ่านและยอมรับ [ข้อตกลงและเงื่อนไขการใช้งาน](terms.md) ก่อนการใช้งาน.

---

**ผู้จัดทำ**  
OTRONTO
