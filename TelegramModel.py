from datetime import datetime
import time
import telegram


class TelegramModel():
    
    def __init__(self):
        self.chat_token = "5294332221:AAEllq5Bw4CSFa3WzP1VB-5HfaNziKk7ypc"
        self.bot = telegram.Bot(token = self.chat_token)
        self.user_lst = []
    
    def get_user(self): #유저 업데이트 하는 함수
        updates = self.bot.getUpdates()
        user_set = set()
        for u in updates:
            user_set.add(u.message['chat']['id'])
            self.user_lst = list(user_set)

    def send_message(self,text): #메세지 보내는 함수
        self.get_user()
        for usr in self.user_lst:
            self.bot.sendMessage(chat_id = usr, text=text)
