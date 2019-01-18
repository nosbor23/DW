import smtplib
from email.MIMEMultipart import MIMEMultipart
from email.MIMEText import MIMEText
from email.MIMEBase import MIMEBase
from email import encoders
 
fromaddr = "lasos.testes@gmail.com"
toaddr = "robsonviei@hotmail.com"
 
msg = MIMEMultipart()
 
msg['From'] = fromaddr
msg['To'] = toaddr
msg['Subject'] = "Testes computacionais"
 
body = "Em anexo"
 
msg.attach(MIMEText(body, 'plain'))
 
filename = "resultados.zip"
attachment = open("C:\\Users\\Robson\\Desktop\\Robson\\Teste Robson 1902\\resultados.zip", "rb")
 
part = MIMEBase('application', 'octet-stream')
part.set_payload((attachment).read())
encoders.encode_base64(part)
part.add_header('Content-Disposition', "attachment; filename= %s" % filename)
 
msg.attach(part)
 
server = smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()
server.login(fromaddr, "tcp35176")
text = msg.as_string()
server.sendmail(fromaddr, toaddr, text)
server.quit()
