import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from spyrl.listener.session_listener import SessionListener
from spyrl.listener.trial_listener import TrialListener
from spyrl.util.util import override
from docutils.parsers.rst.directives import body

''' Send email through a gmail account when learning has started and when learning has finished.
    For this to be successful, the Gmail account must be configured to allow sending emails from non-gmail interfaces 
    The gmail account, password and the target email must be stored in a spyrl.gmail.config file in the user's home directory.
    This setting is meant so that no gmail / password is checked in to a source control.
    The spyrl.gmail.config file must have these 3 lines, such as these:
    
    useraccount@gmail.com
    gmail_password
    target@example.com
    '''

class Gmailer(TrialListener, SessionListener):
    def __init__(self, session_name, subject=None, body=None):
        self.session_name = session_name
        self.subject = subject
        self.body = body
        self.ok = False

    @override(SessionListener)
    def before_session(self, event):
        from pathlib import Path
        import os
        home_dir = str(Path.home())
        config_file = os.path.join(home_dir, 'spyrl.gmail.config')
        if not os.path.exists(config_file):
            print("spyrl.gmail.config cannot be found in ", home_dir)
            return
        f = open(config_file, "r")
        self.gmail = f.readline().rstrip()
        self.password = f.readline().rstrip()
        self.to = f.readline().rstrip()
        f.close()
        if self.to == '':
            print('Invalid spyrl.gmail.config')
            return
        try:
            self.body = self.session_name + ' earning has started.'
            self.subject = self.session_name + ' learning has started.'
            self.send_email()            
        except:
            print("Cannot send email")
        finally:
            self.ok = True
            print("Gmailer seems to be working")

    @override(SessionListener)
    def after_session(self, event):
        try:
            self.body = self.session_name + ' earning has finished.'
            self.subject = self.session_name + ' learning has finished.'
            self.send_email()
        except:
            print("Cannot send email")
        
    def send_email(self):
        if not self.ok:
            return
        mail_content = self.body
        sender_address = self.gmail
        message = MIMEMultipart()
        message['From'] = self.gmail
        message['To'] = self.to
        message['Subject'] = self.subject
        message.attach(MIMEText(mail_content, 'plain'))
        #Create SMTP session for sending the mail
        session = smtplib.SMTP('smtp.gmail.com', 587) #use gmail with port
        session.starttls() #enable security
        session.login(self.gmail, self.password) #login with mail_id and password
        text = message.as_string()
        session.sendmail(sender_address, self.to, text)
        session.quit()