
#import flask
from flask import Flask, request, render_template, redirect, url_for, jsonify , flash
from flask_login import LoginManager, current_user, login_user, logout_user, login_required, UserMixin
import yfinance as yf
#from yfinance.exceptions import YFChartError

from chiave_segreta import chiave
import json
from sklearn.linear_model import LinearRegression
from joblib import dump,load
import numpy as np
import pandas as pd
import login
from google.cloud import firestore,storage
from google.cloud.firestore_v1 import ArrayUnion
from datetime import datetime
from datetime import timedelta
import os
import openpyxl


class User(UserMixin):
    def __init__(self, username):
        self.id = username
        self.username = username
        self.stocks=[]

app = Flask(__name__, static_folder='assets')
app.config['SECRET_KEY'] = chiave

login_manager = LoginManager(app)

login_manager.login_view='/templates/login.html'

#users_db = {'gigi@gigi': {'password': 'sium', 'stocks': []}}

dbName = 'a1234'
collUsers = 'utenti'
stockDB = firestore.Client.from_service_account_json('pc2024-427710-8d91144a6a89.json', database=dbName) #fare qualcosa

####NUOVA PARTE
dbStock_list='azioni'
collAzioni = 'stocks'

DBazioni=firestore.Client.from_service_account_json('pc2024-427710-8d91144a6a89.json', database=dbStock_list)

usersDB = {}

DBstock_list={}

storage_client = storage.Client.from_service_account_json('pc2024-427710-8d91144a6a89.json')  # accedo al cloud storage


def get_historical_data(ticker):
    docRef = DBazioni.collection(collAzioni).document(ticker).collection('daily_data')
    query = docRef.order_by('date', direction=firestore.Query.ASCENDING)
    results = query.stream()
    data_list = []
    for doc in results:
        data = doc.to_dict()
        data['date']=data['date'].date()
        data_list.append(data)
    df = pd.DataFrame(data_list)
    return df

def is_valid_ticker(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        if info and 'symbol' in info:
            return True
    except (ValueError, KeyError):
        pass
    return False

def get_last_date(ticker):
    docRef = DBazioni.collection(collAzioni).document(ticker).collection('daily_data')
    query = docRef.order_by('date', direction=firestore.Query.DESCENDING).limit(1)
    results = query.stream()
    last_date = None
    for doc in results:
        last_date = doc.id
    return last_date

def save_stock_data_batch(ticker, data):
    docRef = DBazioni.collection(collAzioni).document(ticker)
    batch = DBazioni.batch()
    data = data.set_index('Date')
    for date, row in data.iterrows():
        date_str = date.strftime('%Y-%m-%d')
        doc_date_ref = docRef.collection('daily_data').document(date_str)

        batch.set(doc_date_ref, {
            'date': date,
            'open': row['Open'],
            'high': row['High'],
            'low': row['Low'],
            'close': row['Close'],
            'volume': row['Volume']
        })
    batch.commit()
    print(f"Salvataggio dei dati di {ticker} completato con successo.")

def save_model(model):
    file_name = model
    bucket_name = 'bucket_ml_models'
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    blob.upload_from_filename(file_name)
    os.remove(file_name)

def get_model(model):
    bucket_name='bucket_ml_models'
    bucket=storage_client.bucket(bucket_name)  # scelgo il bucket
    blob = bucket.blob(model)  # assegno il nome del file di destinazione
    blob.download_to_filename(model)
    Model=load(model)
    return Model

def make_prediction(df,ticker):
    time_window=60
    horizon=30
    model=get_model(f'MODELLO_univariato_{ticker}.joblib')
    #model=load(f'MODELLO_univariato_{ticker}.joblib')
    df_pred = df[-(time_window + horizon):]
    df_pred = df_pred.reset_index()
    X = []
    previsions_days=[]
    for i in range(len(df_pred) - time_window + 1):
        #a = df_pred['Close'].iloc[i:i + time_window]
        #last_date=df_pred['Date'].iloc[time_window+i-1]
        a = df_pred['close'].iloc[i:i + time_window]
        last_date = df_pred['date'].iloc[time_window + i - 1]
        #print(a,last_date)
        giorni_da_aggiungere = horizon
        giorni_aggiunti = 0
        while giorni_aggiunti < giorni_da_aggiungere:
            last_date += timedelta(days=1)
            if last_date.weekday() < 5:
                giorni_aggiunti += 1
        prev_date = last_date
        previsions_days.append(prev_date)
        feature = a.values.tolist()
        X.append(feature)
    X = np.array(X)

    prev = []
    for i in range(len(X)):
       # print(X[i])
        y_p = model.predict([X[i]])
        #print(y_p)
        prev.append(y_p[0])

    return prev,previsions_days

def getUsersDB():
    usersList = stockDB.collection(collUsers).stream()
    usersDB = {user.to_dict()["username"]: {"password": user.to_dict()["password"],
                                            "email": user.to_dict()["email"],
                                            "stocks_list":user.to_dict()["stocks"]} for user in usersList}
    return usersDB


def updateUsersDB(username, password, email,stocks_list,dn,usersDB):
    docVal = {}
    docVal["username"] = username  # aggiungo username
    docVal["password"] = password  # aggiungo password
    docVal["data nascita"]=dn
    docVal["email"] = email  # aggiungo email
    docVal["stocks"] = stocks_list

    print("docVal: ", docVal)

    docRef = stockDB.collection(collUsers).document(username)  # imposto il documento
    docRef.set(docVal)  # e lo scrivo

    usersDB[username] = {"password": password, "email": email, "stocks":stocks_list}
    usersDB = getUsersDB()  # riacquisco il DB completo (più istanze contemporaneamente possibili)
    return usersDB

def get_user_stocks(user_id):
    doc_ref = stockDB.collection('utenti').document(user_id)
    doc = doc_ref.get()
    if doc.exists:
        return doc.to_dict().get('stocks', [])
    return []

@login_manager.user_loader  # carico il nome dell'utente loggato
def load_user(username):  # ritorno nome utente se in db altrimenti None
    usersDB = getUsersDB()  # acquisisco i dati degli utenti registrati
    if username in usersDB:
        return User(username)
    return None

@app.route('/')
def main():
    return render_template('login.html')

@app.route('/utente_loggato')
@login_required
def index():
    return render_template('finance_tracker.html',username=current_user.username)

@app.route('/registrazione', methods=['GET', 'POST'])
def registrazione():
    if request.method == 'POST':
        if current_user.is_authenticated:
            return jsonify({'success': False, 'message': 'Sei già autenticato. Vai al menu generale.'})

        username = request.values['username']
        email = request.values['email']
        password = request.values['password']
        dn = request.values['data_nascita']
        stocks_list = []

        usersDB = getUsersDB()

        if username == "" or password == "" or email == "":
            return jsonify({'success': False, 'message': 'Compila tutti i campi!'})

        if email in [valDict["email"] for valDict in usersDB.values()]:
            return jsonify({'success': False, 'message': 'Email già in uso!'})

        if username in usersDB:
            return jsonify({'success': False, 'message': 'Username già esistente!'})

        usersDB = updateUsersDB(username, password, email, stocks_list, dn, usersDB)

        return jsonify({'success': True, 'message': 'Registrazione completata con successo!'})

    return render_template('registrazione.html')


@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('main'))

@app.route('/login', methods=['POST'])
def login():
    if request.method == 'POST':
        if current_user.is_authenticated:
            return redirect(url_for('index'))
            #return render_template('finance_tracker.html', username=current_user.username)
            #return redirect(url_for('/utente_loggato'))
            #return redirect("/templates/finance_tracker.html")

            #return render_template('/templates/finance_tracker.html', username=current_user.username)
            #return redirect(url_for('finance_tracker'))

        username = request.values['username']
        password = request.values['password']

        usersDB = getUsersDB()
        #print(usersDB)
        #current_user.stocks = usersDB[username]["stocks_list"]
        #print('asksakks',current_user.stocks)

        if username in usersDB and password == usersDB[username]["password"]:
            #print(usersDB)
            #current_user.stocks = usersDB[username]["stocks_list"]
            #print('asksakks', current_user.stocks)

            login_user(User(username))
            current_user.stocks = get_user_stocks(current_user.id)

            for s in current_user.stocks:
                last_date = get_last_date(s)
                last_date = datetime.strptime(last_date, "%Y-%m-%d")
                today=datetime.now()
                if today > last_date:
                    df = yf.download(s, start=last_date, end=today).reset_index()
                    save_stock_data_batch(s,df)

            return redirect(url_for('index'))
        else:
            return jsonify({'message': 'Errore!'})

        #flash('Invalid username or password', 'error')
        #return redirect(url_for('login'))

    #return render_template('login.html')

@app.route('/lista_titoli')
@login_required
def lista_titoli():
    #print('ff',current_user.stocks)
    #current_user.stocks=get_user_stocks(current_user.id)
    print(usersDB)
    current_user.stocks=get_user_stocks(current_user.id)
    #print(stock_list)
    return render_template('lista_titoli.html', user_stocks=current_user.stocks,username=current_user.username)

@app.route('/add_stock', methods=['POST'])
@login_required
def add_stock():
    user_stocks=get_user_stocks(current_user.username)
    stock=request.form.get('stock')
    stock=stock.upper()
    #df=yf.download(stock,period='1y').reset_index()
    #df=get_historical_data(stock)[:252].reset_index()

    try:
        if not is_valid_ticker(stock):
            return jsonify({'message':'Titolo non valido'})
        df=get_historical_data(stock)[-252:].reset_index()
        #print(len(df))
        #df=get_historical_data(stock)
        doc_ref = stockDB.collection('utenti').document(current_user.username)
        time_window = 60
        horizon = 30
        X = []
        y = []
        for i in range(time_window, df['close'].shape[0] - horizon):
            X.append(df['close'][i - time_window:i])
            y.append(df['close'][i + horizon])
        X = np.array(X)
        y = np.array(y)
        lr = LinearRegression()
        lr.fit(X, y)
        dump(lr,f'MODELLO_univariato_{stock}.joblib')
        save_model(f'MODELLO_univariato_{stock}.joblib')

        #if df.empty or " " in stock or stock in current_user.stocks:
        #    return jsonify({'message': 'Titolo non valido.'})
        #else:
        doc_ref.update({'stocks': ArrayUnion([stock])})
        #print(user_stocks)
        if stock not in user_stocks:
            return jsonify({'message': 'Titolo aggiunto con successo!'})
        else:
            return jsonify({'message': f'Hai aggiornato i dati di training del modello per il titolo {str(stock)}'})

    except KeyError as e:
        return jsonify({'message': f"Il titolo {str(stock)} non è presente nel DB, assicurati di controllare prima lo storico cliccando su 'CERCA'"})

@app.route('/finance_tracker', methods=['POST'])
@login_required
def cerca_titolo():
    if request.method == 'POST':
        ticker = request.form.get('stock')
        ticker = ticker.upper()
        print(ticker)

        if is_valid_ticker(ticker):
            last_date = get_last_date(ticker)
            if last_date is None:
                df = yf.download(ticker, period='5y').reset_index()
                if df.empty:
                    return jsonify({'success': False, 'message': 'Titolo non valido.'})
                else:
                    save_stock_data_batch(ticker, df)
            else:
                last_date = datetime.strptime(last_date, "%Y-%m-%d")
                today = datetime.now()
                if today > last_date:
                    df = yf.download(ticker, start=last_date, end=today).reset_index()
                    if df.empty:
                        return jsonify({'success': False, 'message': 'Titolo non valido.'})
                    else:
                        save_stock_data_batch(ticker, df)
            historical_data = get_historical_data(ticker)
            historical_data['date'] = pd.to_datetime(historical_data['date'])
            historical_data['date'] = historical_data['date'].dt.strftime('%Y-%m-%d')
            historical_data = historical_data[['date', 'close']]
            data = historical_data.to_dict(orient='records')
            return jsonify({'success': True, 'data': data})
        else:
            return jsonify({'success': False, 'message': 'Titolo non valido.'})

@app.route('/get_forecast', methods=['POST'])
@login_required
def get_stock_forecast():
    if request.method == 'POST':
        ticker = request.form.get('stock')
        df=get_historical_data(ticker).reset_index()
        #df=yf.download(ticker,period='5y').reset_index()
        #dates_hist=df['Date'].to_list()
        #close_values_hist=df['Close'].to_list()
        dates_hist = df['date'].to_list()
        close_values_hist = df['close'].to_list()
        y_pred,dates_prev=make_prediction(df,ticker)
        #y_pred=y_pred.tolist()
        closes=close_values_hist+y_pred
        dates=dates_hist+dates_prev
        forecast=[False for i in range(len(close_values_hist))]
        for i in range(len(y_pred)):
            forecast.append(True)
        new_df = {'Date': dates, 'Close': closes, "Forecast":forecast}
        df1 = pd.DataFrame(new_df)
        data = df1.to_dict(orient='records')
        #message=buy_or_sell(df,y_pred)
        cl_W_H=close_values_hist[-1]
        log_return,d=calculate_LOG_returns(y_pred,cl_W_H)
        if log_return>=0:
            message='Buy stock'
        else:
            message="Don't buy stock"
        #return data,message
        log_return = round(log_return, 2)
        log_return= f'{log_return} %'
        return jsonify({'data': data, 'message': message, 'log_return': log_return, 'd' : d})


def buy_or_sell(df, prevision):
    current_price = df['close'].iloc[-1]
    mean_forecast = np.mean(prevision)
    if mean_forecast > current_price:
        return 'Compra il titolo'
    else:
        return 'Vendi il titolo'

def calculate_LOG_returns(prevision,history):
    data=prevision
    data.insert(0,history)
    df_r = pd.DataFrame(data, columns=['Returns'])
    log_returns = np.log(df_r['Returns']).diff().sum()
    #log_returns=round(log_returns, 2)

    d="""
    Log returns are the logarithmic difference between the price of an asset at two different times.
    They’re a way of measuring the percentage change in the value of an asset over time.
    """
    return log_returns,d

if __name__ == '__main__':
    app.run(debug=True)

