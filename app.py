from flask import Flask,jsonify,render_template,url_for,request
import json
import spacy
from spacy.util import minibatch, compounding
import random
from spacy import displacy
import en_core_web_sm

nlp=spacy.load("en_core_web_sm") 

app = Flask(__name__)
PORT = 3000

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/process',methods = ["GET","POST"])
def process():
    # Getting the ner component
    ner=nlp.get_pipe('ner')

    #New label to add
    LABEL = ["FOOD","ELECTRONICS","DRESS","TOYS","BEAUTY",
    "HOME UTILITIES","BEAUTY","CAR"]

    for label in LABEL:
        ner.add_label(label)
    
    # Training examples in the required format
    TRAIN_DATA =[ ("Pizza is a common fast food.", {"entities": [(0, 5, "FOOD")]}),
              ("Pasta is an italian recipe", {"entities": [(0, 5, "FOOD")]}),
              ("China's noodles are very famous", {"entities": [(8,14, "FOOD")]}),
              ("India's Biryani are very famous", {"entities": [(8,14, "FOOD")]}),
              ("Shrimps are famous in China too", {"entities": [(0,4, "FOOD")]}),
              ("Rice are famous in India", {"entities": [(0,7, "FOOD")]}),
              ("Lasagna is another classic of Italy", {"entities": [(0,7, "FOOD")]}),
              ("Sushi is extemely famous and expensive Japanese dish", {"entities": [(0,5, "FOOD")]}),
              ("Unagi is a famous seafood of Japan", {"entities": [(0,5, "FOOD")]}),
              ("Tempura , Soba are other famous dishes of Japan", {"entities": [(0,7, "FOOD")]}),
              ("Burgers are the most commonly consumed fastfood", {"entities": [(0,7, "FOOD")]}),
              ("Udon is a healthy type of noodles", {"entities": [(0,4, "FOOD")]}),
              ("Chocolate soufflé is extremely famous french cuisine", {"entities": [(0,17, "FOOD")]}),
              ("Flamiche is french pastry", {"entities": [(0,8, "FOOD")]}),
              ("Burgers are the most commonly consumed fastfood", {"entities": [(0,7, "FOOD")]}),
              ("Frenchfries are considered too oily", {"entities": [(0,11, "FOOD")]}),
              ("Cookies are very tasty", {"entities": [(0,7, "FOOD")]}),
              ("Bread butter jam is favorite trio", {"entities": [(0,16, "FOOD")]}),
              ("Dark chocolate is good for heart", {"entities": [(0,14, "FOOD")]}),
              ("Steamed food are the best", {"entities": [(0,12, "FOOD")]}),
              ("Momos is native to Tibet and Nepal", {"entities": [(0,5, "FOOD")]}),
              ("Cheese is a dairy product derived from milk", {"entities": [(0,6, "FOOD")]}),
              ("Luqaimat is made of dumplings", {"entities": [(0,8, "FOOD")]}),
              ("The national fruit of the UAE is the dates", {"entities": [(12,33, "FOOD")]}),
              ("Apple Pie is famous in America", {"entities": [(0,9, "FOOD")]}),
              ("Spicy chicken sandwich is famous in America", {"entities": [(0,21, "FOOD")]}),
              ("Gyoza is popular dish in Japan", {"entities": [(0,5, "FOOD")]}),
              ("The national dish of the Japan is the Curry Rice", {"entities": [(30,40, "FOOD")]}),
              ("The most popular in Antarctica is definitely duck", {"entities": [(38,43, "FOOD")]}),
              ("African meal is made with starchy items and herbs", {"entities": [(0,11, "FOOD")]}),
              ("Laptops are getting more powerful", {"entities": [(0,7, "ELECTRONICS")]}),
              ("Wireless earphones are becoming popular", {"entities": [(0,17, "ELECTRONICS")]}),
              ("Fridge can be ordered in online ", {"entities": [(0,6, "ELECTRONICS")]}),
              ("Camera click pictures", {"entities": [(0,6, "ELECTRONICS")]}),
              ("I rented a screwdriver from our neighbour", {"entities": [(11,22, "ELECTRONICS")]}),
              ("Computers are really fast", {"entities": [(0,9, "ELECTRONICS")]}),
              ("Clock shows time", {"entities": [(0,5, "ELECTRONICS")]}),
              ("Wireless mouse is handy", {"entities": [(0,15, "ELECTRONICS")]}),
              ("Powerbank is useful", {"entities": [(0,9, "ELECTRONICS")]}),
              ("My phone works so well", {"entities": [(0,8, "ELECTRONICS")]}),
              ("Laptops are handy computers", {"entities": [(0,7, "ELECTRONICS")]}),
              ("A CPU is a central processing unit that executes instructions from a computer program", {"entities": [(0,5, "ELECTRONICS")]}),
              ("Mobile phone is a portable telephone", {"entities": [(0,12, "ELECTRONICS")]}),
              ("Tablet is a wireless, portable personal computer", {"entities": [(0,6, "ELECTRONICS")]}),
              ("Telephone is a telecommunications device", {"entities": [(0,9, "ELECTRONICS")]}),
              ("Digital thermometer is used now", {"entities": [(0,19, "ELECTRONICS")]}),
              ("Blender is a appliance used to crush food", {"entities": [(0,7, "ELECTRONICS")]}),
              ("Radio is used to play songs", {"entities": [(0,5, "ELECTRONICS")]}),
              ("Computer as powerful as the human brain", {"entities": [(0,8, "ELECTRONICS")]}),
              ("Computer is a machine that can be programmed to complete both simple and complex tasks", {"entities": [(0,8, "ELECTRONICS")]}),
              ("Desktop computer is a personal computing device", {"entities": [(0,16, "ELECTRONICS")]}),
              ("The first computer looked like an oversized calculator", {"entities": [(10,19, "ELECTRONICS")]}),
              ("Fan is in offer", {"entities": [(0,3, "ELECTRONICS")]}),
              ("Laptops are compact in size", {"entities": [(0,7, "ELECTRONICS")]}),
              ("Pendrive is used to store data", {"entities": [(0,8, "ELECTRONICS")]}),
              ("Electric Kettle keeps the water hot", {"entities": [(0,12, "ELECTRONICS")]}),
              ("Air Purifier purifies air", {"entities": [(0,21, "ELECTRONICS")]}),
              ("Water Purifier purifies water", {"entities": [(0,15, "ELECTRONICS")]}),
              ("Headphones can damage your ears", {"entities": [(0,10, "ELECTRONICS")]}),
              ("Sewing Machine helps to stitch clothes", {"entities": [(0,14, "ELECTRONICS")]}),
              ("Generators are useful equipment that generates electricity ", {"entities": [(0,10, "ELECTRONICS")]}),
              ("Washing Machine helps to wash clothes", {"entities": [(0,16, "ELECTRONICS")]}),
              ("Headset is a headphone combined with a microphone", {"entities": [(0,7, "ELECTRONICS")]}),
              ("Iron box helps to iron clothes", {"entities": [(0,8, "ELECTRONICS")]}),
              ("Oven is used to bake cakes", {"entities": [(0,4, "ELECTRONICS")]}),
              ("Microphone are used to increase the volume of the human voice", {"entities": [(0,10, "ELECTRONICS")]}),
              ("Cashmere wool is soft", {"entities": [(0,5, "DRESS")]}),
              ("Cotton to silk", {"entities": [(0,6, "DRESS")]}),
              ("Silk dresses are fancy", {"entities": [(0,4, "DRESS")]}),
              ("Clay colours are bright and lovely", {"entities": [(0,4, "TOYS")]}),
              ("Soft toys are best to play with", {"entities": [(0,9, "TOYS")]}),
              ("Lakme lipstick quality is good", {"entities": [(0,5, "BEAUTY")]}),
              ("Lipstick add colours to lips", {"entities": [(0,8, "BEAUTY")]}),
              ("Mascara is adds beauty to eyes", {"entities": [(0,7, "BEAUTY")]}),
              ("Mattress is soft and comfy", {"entities": [(0,8, "HOME UTILITIES")]}),
              ("Mattress helps in good sleep", {"entities": [(0,8, "HOME UTILITIES")]}),
              ("Table helps in dining", {"entities": [(0,5, "HOME UTILITIES")]}),
              ("Walmart is a leading e-commerce company", {"entities": [(0, 7, "ORG")]}),
              ("BMW is costly", {"entities": [(0,3, "CAR")]}),
              ("I ordered this from ShopClues", {"entities": [(20,29, "ORG")]}),
              ("Washer is useful", {"entities": [(0,6, "HOME UTILITIES")]}),
              ("Table is made of wood", {"entities": [(0,5, "HOME UTILITIES")]}),
              ("Flipkart started it's journey from zero", {"entities": [(0,8, "ORG")]}),
              ("I recently ordered from Max", {"entities": [(24,27, "ORG")]}),
              ("Flipkart is recognized as leader in market",{"entities": [(0,8, "ORG")]}),
              ("I recently ordered from Swiggy", {"entities": [(24,29, "ORG")]}) 
             
           ]
    # Resume training
    optimizer = nlp.resume_training()
    move_names = list(ner.move_names)

    # List of pipes you want to train
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]

    # List of pipes which should remain unaffected in training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

    # Begin training by disabling other pipeline components
    with nlp.disable_pipes(*other_pipes) :
        sizes = compounding(1.0, 4.0, 1.001)
        # Training for 30 iterations     
        for itn in range(30):
        # shuffle examples before training
            random.shuffle(TRAIN_DATA)
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAIN_DATA, size=sizes)
            #Dictionary to store losses
            losses = {}
            for batch in batches:
                texts, annotations = zip(*batch)
                # Calling update() over the iteration
                nlp.update(texts, annotations, sgd=optimizer, drop=0.35, losses=losses)
                #print("Losses", losses)

                print("processed")

    if request.method == 'POST':
        rawtext = request.form['rawtext']
        doc = nlp(rawtext)

        print("Entities in '%s'" % rawtext)
        d = []

        for ent in doc.ents:
            d.append((ent.label_, ent.text))
            results = d
		    

    return render_template("index.html",results=results)

 
if __name__ == "__main__":
    app.run(debug=True,host="0.0.0.0",port=PORT)