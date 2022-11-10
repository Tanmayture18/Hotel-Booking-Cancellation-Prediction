# We are going to deploy our model using Flask
from flask import Flask,render_template,request
import pickle


app = Flask(__name__)
file=open('model.pk1','rb')
clf=pickle.load(file)
file.close()

@app.route("/",methods=["GET","POST"])
def hello_world():
    if request.method=="POST":
        mydict=request.form
        lead_time=int(mydict['lead_time'])
        adults=int(mydict['adults'])
        children=int(mydict['children'])
        is_repeated_guest=int(mydict['is_repeated_guest'])
        previous_cancellations=int(mydict['previous_cancellations'])
        meal=int(mydict['meal'])
        deposit_type=int(mydict['deposit_type'])
        required_car_parking_spaces=int(mydict['required_car_parking_spaces'])
        customer_type=int(mydict['customer_type'])

        inputfeatures=[lead_time,adults,children,is_repeated_guest,previous_cancellations,meal,deposit_type,required_car_parking_spaces,customer_type]
    
        infprob=clf.predict([inputfeatures])

        return render_template('show.html',inf=infprob)
    return render_template('index.html')

if __name__=='__main__':
    app.run(debug=True)   
    

        
