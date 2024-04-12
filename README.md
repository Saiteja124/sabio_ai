<h2>Deploying ML Model using Flask</h2>

This is a Flask-based web application that allows users to register, login, and upload PDF invoices for processing. The project is about extracting the text from images,scanned documents or pdfs and converting into machine readable text

<h3>Prerequisites</h3>
pip install -r requirements.txt

<h3>Project Structure</h3>
This project has four major parts :

1. model.py - This contains code to extract the text data from images,scanned documents or pdfs.
2. app.py - This contains Flask APIs to instegrate model with webpage
3. templates - This folder contains the HTML template (index.html) to allow user to sign up and sign in.
4. static - This folder contains the css folder with styles and scripts and images.

<h3>Running the project</h3>
1. Ensure that you are in the project home directory.

Run python app.py


By default, flask will run on port 5000.
Navigate to URL http://127.0.0.1:5000/ (or) http://localhost:5000
You should be able to view the homepage. Enter a valid username and email to sign in for extracting the text



