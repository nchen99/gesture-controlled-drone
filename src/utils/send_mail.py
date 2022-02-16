import smtplib, ssl
from os.path import basename
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formatdate

port = 465
sender = "docsfocsbocs@gmail.com"
# Email password here
password = "Capstone21"
context = ssl.create_default_context()


def send_mail(send_to, subject, text, files=None):
    global port, sender, password, context

    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = send_to
    msg['Date'] = formatdate(localtime=True)
    msg['Subject'] = subject

    msg.attach(MIMEText(text))

    for f in files or []:
        with open(f, "rb") as fil:
            part = MIMEApplication(
                fil.read(),
                Name=basename(f)
            )
        # After the file is closed
        part['Content-Disposition'] = 'attachment; filename="%s"' % basename(f)
        msg.attach(part)

    with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
        server.login(sender, password)
        server.sendmail(sender, send_to, msg.as_string())

# example:
# send_mail("shawnlu4@gmail.com", "Test", "test", files=["./file.py"])