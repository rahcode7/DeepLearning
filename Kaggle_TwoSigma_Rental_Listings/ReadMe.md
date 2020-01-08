# Two Sigma Rental Challenge

#### Clone repository 

## Step 1 - Data Analysis  & Modelling

##### It contains 2 folders 
	1.0 cd rental_listings

	1.1 python Rental_classifier_LGBMCat.py
		
		Outputs - 
		a.restapi/model_rental_lgbmcat.pkl 
		b.restapi/model_rental_lgbmcat_columns.pkl

## Step 2 - Serving the model via flask api 

#### 1. Set up and run server  

##### Run the following commands in terminal
	1.1 pip install flask      
	1.2 cd restapi            
	1.3 python flask_serving.py    

    It Opens up model serving at following url - localhost:5000/predict   

#### 2. Test sample payload

#### Download postman for testing   
	https://www.getpostman.com/downloads/


##### Paste sample payload as a POST request body

[{"bathrooms":1.0,"bedrooms":1,"building_id":"79780be1514f645d7e6be99a3de696c5","created":"2016-06-11 05:29:41","description":"Large with awesome terrace--accessible via bedroom and living room. Unique find in the LES.Apartment Features:-Large terrace via bedroom and living room-Hardwood floors-Newly renovated -Granite counter top-Breakfast Bar-Ample counter space and storage-Dishwasher-Great Lighting Neighborhood Features:-A few blocks from Whole Foods-1 block from the J, Z and M subway-All the restaurants and night life the Lower East Side is known for (Hotel Chantel, DL, Pianos)Call\\/txt\\/Email James to set up a showing:kagglemanager@renthop.com<br \\/><br \\/><br \\/><br \\/><br \\/><br \\/><p><a  website_redacted ","display_address":"Suffolk Street","features":["Elevator","Laundry in Building","Laundry in Unit","Dishwasher","Hardwood Floors","Outdoor Space"],"latitude":40.7185,"listing_id":7142618,"longitude":-73.9865,"manager_id":"b1b1852c416d78d7765d746cb1b8921f","photos":["https:\\/\\/photos.renthop.com\\/2\\/7142618_1c45a2c8f45e649b9ee77681cc7ca438.jpg","https:\\/\\/photos.renthop.com\\/2\\/7142618_2a0268ff01f834c1039027a04e54edf4.jpg","https:\\/\\/photos.renthop.com\\/2\\/7142618_1645edaeb3892d35c190356eeb16bd75.jpg","https:\\/\\/photos.renthop.com\\/2\\/7142618_ca5c03339bd1f021b94da72af7356bca.jpg","https:\\/\\/photos.renthop.com\\/2\\/7142618_b129d432a96a0ad419f1af430f4a20ff.jpg","https:\\/\\/photos.renthop.com\\/2\\/7142618_dd3c3651b991455d3ed7403766c6941d.jpg","https:\\/\\/photos.renthop.com\\/2\\/7142618_4ddef2aee0c343f5a86da7113f9336fc.jpg","https:\\/\\/photos.renthop.com\\/2\\/7142618_6c51aec64570affecc573efbdc4453ca.jpg"],"price":2950,"street_address":"99 Suffolk Street”}
]


##### Output Predictions - Probabilities (In order of)- high,medium,low
See screenshot - postman.png
