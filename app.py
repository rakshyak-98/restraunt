from flask import Flask, request, render_template, redirect, session
from flask_sqlalchemy import SQLAlchemy
import bcrypt
import main as m
import main1 as m2
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)
app.secret_key = 'secret_key'


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))

    def __init__(self, email, password, name):
        self.name = name
        self.email = email
        self.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password.encode('utf-8'))


with app.app_context():
    db.create_all()


@app.route('/')
def index():
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # handle request
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        
        
        new_user = User(name=name, email=email, password=password)
        try:
            db.session.add(new_user)
            db.session.commit()
        except Exception as error:
            return render_template('login.html',script = "unique()")
        return redirect('/login')

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email).first()

        if user and user.check_password(password):
            session['email'] = user.email
            return redirect('/dashboard')
        else:
            return render_template('login.html', script='absent()')

    return render_template('login.html')


@app.route('/dashboard', methods=["GET", "POST"])
def dashboard():
    if session['email']:
        user = User.query.filter_by(email=session['email']).first()
        return render_template('dashboard.html', user=user)
    
    return redirect('/login')


@app.route('/logout')
def logout():
    session.pop('email', None)
    return redirect('/login')

@app.route('/location')
def location():
    if session['email']:
        user = User.query.filter_by(email=session['email']).first()
        # return render_template('locationbased.html', user=user)
        return render_template('newlocation.html', user=user)
    
    return redirect('/login')
@app.route('/rating')
def rating():
    if session['email']:
        user = User.query.filter_by(email=session['email']).first()
        return render_template('rating.html', user=user)
    
    return redirect('/login')
@app.route('/cuisine')
def cuisine():
    if session['email']:
        user = User.query.filter_by(email=session['email']).first()
        return render_template('cuisine.html', user=user)
    
    return redirect('/login')

@app.route('/similarrestaurent')
def similarrestaurent():
    if session['email']:
        user = User.query.filter_by(email=session['email']).first()
        return render_template('similarrestaurent.html', user=user)
    
    return redirect('/login')

@app.route('/locationsearch', methods=["GET", "POST"])
def location_based():
    # Get the city code from the query parameter
    location= request.form.get('location')
    a = m.locationbased(location)
    # return render_template("city_hotels.html", prediction_text = "Diabetics {}".format(a))  
    if session['email']:
        user = User.query.filter_by(email=session['email']).first()
        return render_template('table.html', user=user, table=a, prediction_text=format(location))
 
@app.route('/city', methods=["GET", "POST"])
def city_based():
    # Get the city code from the query parameter
    city= request.form.get('city')
    a = m.citybased(city)
    # return render_template("city_hotels.html", prediction_text = "Diabetics {}".format(a))  
    if session['email']:
        user = User.query.filter_by(email=session['email']).first()
        return render_template('table.html', user=user, table=a, prediction_text=format(city))
 


@app.route('/similar', methods=["GET", "POST"])
def similar():

    similar = request.form.get('similar')
    a = m2.recommend(similar)

    if session['email']:
        user = User.query.filter_by(email=session['email']).first()
        return render_template('table2.html', user=user, table=a, prediction_text=format(similar))
    

@app.route('/rate', methods=["GET", "POST"])
def rate():
    rate = request.form.get('rate')
    a = m2.predict_rating_for_restaurant(rate)
    if session['email']:
        user = User.query.filter_by(email=session['email']).first()
        return render_template('prediction.html', user=user,prediction_text=format(rate) , prediction_rating=format(a))

@app.route('/cuisinebased', methods=["GET", "POST"])
def cuisinebased():
    cuisine = request.form.get('cuisine')
    a = m.cuisinesbased(cuisine)
    if session['email']:
        user = User.query.filter_by(email=session['email']).first()
        return render_template('table1.html', user=user, table=a, prediction_text=format(cuisine))
   


if __name__ == '__main__':
    app.run(debug=True)

def findBthSmallest (A, B):
  # Import heapq module to use min-heap
  import heapq
  # Sort the array A in ascending order
  A.sort()
  # Initialize an empty min-heap to store the sums and indices
  heap = []
  # Loop over the array and generate the initial sums
  for i in range(len(A) - 2):
    left = i + 1
    right = len(A) - 1
    sum = A[i] + A[left] + A[right]
    heapq.heappush(heap, (sum, i, left, right))
    right -= 1
  # Initialize a counter to keep track of how many elements we have popped from the heap
  count = 0
  # Repeat until we find the Bth smallest element or the heap is empty
  while heap and count < B:
    # Pop the top element of the heap and store it in ans
    ans, i, left, right = heapq.heappop(heap)
    # Increment the counter by one
    count += 1
    # If the counter is equal to B, we have found the answer and we can return it
    if count == B:
      return ans
    # Otherwise, we need to generate more sums from the indices of the popped element
    # If left + 1 < right, we can increment the left pointer by one and push the new sum to the heap
    if left + 1 < right:
      left += 1
      sum = A[i] + A[left] + A[right]
      heapq.heappush(heap, (sum, i, left, right))
  # If the heap is empty and we have not found the Bth smallest element, we can return -1
  return -1
