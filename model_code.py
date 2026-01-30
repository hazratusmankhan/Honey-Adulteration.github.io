from flask import Flask, request, render_template
import io
import csv
import pandas as pd
import pickle

app = Flask(__name__, template_folder='templates')

# Load models for each task
model_honey_adulteration = pickle.load(open('model_honey_adulteration.pkl', 'rb'))
model_honey_adulterationandclass = pickle.load(open('model_honey_adulteration+class.pkl', 'rb'))
model_honey_class = pickle.load(open('model_honey_class.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/testing.html')
def testing():
    return render_template('testing.html')

def transform(text_file_contents):
    return text_file_contents.replace("=", ",")

@app.route('/transform', methods=["POST"])
def transform_view():
    f = request.files['data_file']
    if not f:
        return "No file provided"

    try:
        stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)
        csv_input = csv.reader(stream)
        data = list(csv_input)
        print(data)  # You can remove this print statement
        stream.seek(0)
        result = transform(stream.read())
        df = pd.read_csv(io.StringIO(result))
        
        # Perform prediction based on the task
        task = request.form['task']
        if task == 'honey_adulteration':
            output = model_honey_adulteration.predict(df)
            return render_template('testing.html', prediction_text='The provided sample is adulterated {}'.format(output))
        elif task == 'model_honey_class':
            output = model_honey_class.predict(df)
            return render_template('testing.html', prediction_text='The Provided honey sample is of type {}'.format(output))
        elif task == 'honey_adulteration and class':
            output = model_honey_adulterationandclass.predict(df)
        
        return render_template('testing.html', prediction_text='The provided honey sample is [''Adult'' ''Type'' : ]  as  {}'.format(output))
    except Exception as e:
        return "An error occurred: {}".format(str(e))

if __name__ == "__main__":
    app.run(debug=True, port=5000)
