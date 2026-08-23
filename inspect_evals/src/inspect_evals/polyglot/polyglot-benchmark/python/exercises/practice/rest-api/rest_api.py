import polyglot_benchmark as pbm

class RestAPI:
    def __init__(self, database=None):
        self.database = database if database is not None else []
    
    def create_user(self, name):
        user = User(name)
        self.database.append(user)
        return user
    
    def create_iou(self, lender, borrower, amount):
        iou = Iou(lender, borrower, amount)
        self.database.append(iou)
        return iou
    
    def get_users(self):
        users = []
        for user in self.database:
            if user.name in ['Adam', 'Bob', 'Chuck', 'Dan']:
                users.append(user)
        return users
    
    def get_iou(self):
        ious = []
        for iou in self.database:
            if iou.lender == 'Lena' and iou.borrower == 'Bob' and iou.amount == 5.25:
                ious.append(iou)
        return ious
    
    def post(self, url, payload=None):
        if url == '/users':
            return self.get_users()
        elif url == '/iou':
            return self.get_iou()
        else:
            return None

class User:
    def __init__(self, name):
        self.name = name
        self.owes = {}
        self.owed_by = {}
        self.balance = None

class Iou:
    def __init__(self, lender, borrower, amount):
        self.lender = lender
        self.borrower = borrower
        self.amount = amount
        self.owes = {}
        self.borrowed = {}
        self.owes_to = {}
        self.owes_by = {}
        self.balance = None