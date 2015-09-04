
# coding: utf-8

# In[1]:

""" Author: Barsha Shrestha 8/24/15

    Objective: Finding the Pearson Correlation Coefficient using Spark in local mode

    This is a python script to analyze all the data as part of a data challenge.
    The challenge was to compute the correlation between food prices & Census income data at a zip-code level, within the city of San Diego.

    There were 3 datasets provided (O, B, C) :
    O = Food price observations. Observations carry latitude, longitude and price, among other columns. 
    They also have a spec_uuid which refers to a specific food product, the meta data about which is in file B below.
    Note that each spec_uuid can have multiple packaging variants, so we have provided the normalized_price and normalized_size_units fields.
    
    B = food products taxonomy. A map from spec_uuid to the bundle.
 Pay special attention to the columns, uuid and bundle e.g. Pork Spare Ribs → Meat

    C = Per zipcode income data from US Census.
    Pay particular attention to the column A00100 (adjusted gross income)
    The data-dictionary is also included as a word doc file (us-income-by-zip-12zdefinition.doc).

    
    I used Spark with Spark SQL for batch processing and making queries/dataframes respectively.
    While Spark isn't a good tool for small data, with a larger dataset and distributed computing, the processing will be much
    faster than with a small dataset.
    
    I used iPython notebook with the databricks package for csv (com.databricks:spark-csv_2.10:1.1.0) along with 4 threads 
    running on my local machine. The entire bash command is below:
    PYSPARK_DRIVER_PYTHON=/Library/Frameworks/Python.framework/Versions/2.7/bin/ipython PYSPARK_DRIVER_PYTHON_OPTS="notebook --no-browser --port=7777" $SPARK_HOME/bin/pyspark --packages com.databricks:spark-csv_2.10:1.1.0 --master local[4]

    For this to run, the jar file that was build from spark-csv github (spark-csv_2.11-1.2.0.jar) needs to be in Spark's home directory too.
    
    I calculated the Pearsons correlation coefficient for each bundle. The result is in a file named 'Pearson_coeff_by_bundle.csv', which is in this folder. 
    
    I also made a sample scatter plot of food metric vs. weighted AGI by each zip code, for each bundle ('Pearson_Correlation_Coeff_Vegetables').

    There is also a sample code adapted from http://www.christianpeccei.com/zipmap/ that helps plot AGI(Adjusted Gross Income) per zip code per bundle or food metric per zip code per bundle (in this case, vegetable) as necessary. It is only a sample to show how the visualizations can be achieved after the data points are obtained. The picture obtained is ‘SD_AGI.png’, also in this folder.
    
    Out of the 14 bundles, 9 of them had a negative coefficient and 5 of them had a positive coefficient.
    
    I made many tables to keep track of what exactly consisted in each dataframe. The code below in its entirety if ran on Terminal using the command posted above, should run all the way to provide the pearson correlation coefficient by zip code, for each bundle (the code for creating sample plot will have to be commented out as that cannot be done in a spark shell, same for zip code plot on the map)
"""

# coding: utf-8
# The code starts here

#If using spark-submit on the shell, get to the Spark directory and use the command:
#./bin/spark-submit --packages com.databricks:spark-csv_2.10:1.1.0 --master local[4] /Users/barshashrestha/Documents/Premise_Results/Premise_DataChallenge_82515.py


#Also uncomment the 5 lines of code below so there is a SparkContext object beforehand: 
#from pyspark import SparkContext, SparkConf
#from pyspark.sql import SQLContext
#conf = SparkConf().setAppName('Premise').setMaster('local')
#sc = SparkContext(conf=conf)
#sqlContext = SQLContext(sc)

# In[11]:

from pyspark.sql import *
from pyspark import SparkContext, SparkConf
from pyspark.sql.types import *


# In[12]:

####################################To create table for AGI###################################################


incomezipRDD = sc.textFile("file:///Users/barshashrestha/Documents/us-income-by-zip-12zpallagi.csv")
incomezipRDD.getNumPartitions()#to get number of partitions
header=incomezipRDD.first() #just get header


# In[13]:

fields = [StructField(field_name, StringType(), True) for field_name in header.split(',')]
schema = StructType(fields) #schema for building dataframe


# In[14]:

#drop header from csv
headerRDD = incomezipRDD.filter(lambda l : "A00100" in l)
incomenohead = incomezipRDD.subtract(headerRDD)


# In[15]:

splitincome = incomenohead.map(lambda l: l.split(",")).map(lambda p: (p[1], int(p[2]), float(p[4]), float(p[11]))) #to split each row then convert fields as needed 


# In[18]:

dfincome = sqlContext.createDataFrame(splitincome, ['STATE', 'zipcode', 'returns', 'AGI']) # make dataframe from schema so as to perform sql queries
dfincome.registerTempTable("incometable") #register as table to make SQL queries


# In[10]:

AGIrdd = sqlContext.sql("SELECT zipcode, returns, AGI from incometable where STATE = 'CA'") #rdd with only required columns
AGIrdd.registerTempTable("AGItable")


# In[8]:

#load zip codes, map each zip code to filter data for only SD
# The San Diego zip codes were obtained from http://www.sdcourt.ca.gov/pls/portal/docs/PAGE/SDCOURT/GENERALINFORMATION/FORMS/ADMINFORMS/ADM254.PDF and saved to a csv file
zcSDrdd = sc.textFile("file:///Users/barshashrestha/Documents/SC_SD_zipcode .csv")
headerSDrdd = zcSDrdd.filter(lambda l : 'zipcode' in l)
zcSDnohead = zcSDrdd.subtract(headerSDrdd)


# In[9]:

zcSD = zcSDnohead.map(lambda l: l.split(",")).map(lambda p :Row(zipcode = p[0]))
schemazcSD = sqlContext.createDataFrame(zcSD)
schemazcSD.registerTempTable("zcSDTable") #table with only San Diego zip codes

zc = sqlContext.sql("SELECT zipcode from zcSDTable")
zc_only = zc.map(lambda p:  p.zipcode)
zc_only_list = zc_only.collect() #list of all San Diego zip codes




# In[2]:

import sqlite3
from pyzipcode import ZipCodeDatabase
zcdb = ZipCodeDatabase()

#for each zip code in the list 'zc_only_list', get latitude and longitude
lat_list=[]
long_list=[]
for zc in zc_only_list:
    try:
        zip_code = zcdb[zc]
        lat_list.append(str(round(zip_code.latitude,2))) #trying to match to more than 2 decimal places decreased these data points as this lat and long values weren't an exact match with the lat and long provided. Hence some data may have been lost while latitude and longitude matching
        long_list.append(str(round(zip_code.longitude,2)))
        
    except IndexError:
        lat_list.append('200.54') # a latitude and longitude that doesn't actually exist
        long_list.append('200.54')
        

zipped_tuple = zip(zc_only_list,lat_list, long_list)
zip_lat_long_rdd = sc.parallelize(zipped_tuple)

df_zip_lat_long = sqlContext.createDataFrame(zip_lat_long_rdd, ['zipcode', 'latitude', 'longitude'])
df_zip_lat_long.registerTempTable("ziplatlong")
list_ziplatlong =  df_zip_lat_long.collect() #collection of all zip codes with latitude and longitude in San Diego


# In[12]:
#A better and faster way to join these tables would have been to convert the tables to pair RDDs based with the zip code as key and every other value in a tuple. Joining the pair RDDs would take lesser time.
#That would have ensured that each key is in one partition which would make aggregations and joins much faster. But this dataset is small enough for the naive approach to work
df_zc_income_SD = sqlContext.sql("SELECT AGItable.zipcode, returns, AGI from AGItable JOIN zcSDTable on AGItable.zipcode = zcSDTable.zipcode")
df_zc_income_SD.registerTempTable("AGI") #This table has everything needed to calculate AGI


# In[15]:

####################################Table to calculate food price metric#######################################
#Table needs these columns to do all calculations:
#loc_lat, loc_long, spec_uuid, spec_product, normalized_size, normalized_price, size, size_unit, zipcode, avg, demeaned_price#
##################Final Table: Bundle, Zip_Code,, normalized_size, normalized_price, demeaned_price##########


# In[13]:

df_us_taxonomy = sqlContext.read.format('com.databricks.spark.csv').options(header='true').load('/Users/barshashrestha/Documents/Premise/us-sandiego-prices-vs-income/premise-us-food-taxonomy.csv')
df_us_taxonomy.registerTempTable("taxonomy")


# In[14]:

SDobRDD = sc.textFile("file:///Users/barshashrestha/Documents/us-sandiego-observations.csv")
SDobRDD.getNumPartitions() #this is 3

# In[20]:

df_zc_income_SD.printSchema()


# In[16]:
#This is a second load of the same dataset.
observationdf = sqlContext.read.format('com.databricks.spark.csv').options(header='true').load('/Users/barshashrestha/Documents/us-sandiego-observations.csv')
observationdf.registerTempTable("observationdf")


#one of the columns in premise-us-sandiego-observations.csv had a column off by 1 i.e the columns had been shifted by 1. So I had to filter such columns out if I wanted to defined the data type for my fields for my dataframe
cleanob = sqlContext.sql("select loc_lat,loc_long,size, size_units, normalized_price, normalized_size_units, spec_uuid, spec_product from observationdf where normalized_price!='g'")
cleanob.registerTempTable("cleanob")
cleanob.printSchema()

cleanob_rdd = cleanob.rdd



# In[17]:
#method to add zip code to the table with observations based on latitude and longitude
def add_zip_code(row):
    
    zip_code='null'
    
    for eachzll in list_ziplatlong:
        latitude = eachzll[1]
        longitude = eachzll[2]
        
        if row.asDict()["loc_lat"] is not None and row.asDict()["loc_long"] is not None:
            if str(longitude) in row.asDict()["loc_long"] and str(latitude) in row.asDict()["loc_lat"]:
                zip_code= eachzll[0]
                
                break;    
        
    return(row.asDict()["loc_lat"], row.asDict()["loc_long"],row.asDict()["size"],row.asDict()["size_units"],row.asDict()["normalized_price"],row.asDict()["normalized_size_units"],row.asDict()["spec_uuid"], row.asDict()["spec_product"],zip_code)

for eachzll in list_ziplatlong:
    cleanob_rdd_zc = cleanob.map(lambda row: add_zip_code(row))

cleanlistob = cleanob_rdd_zc.collect()


df_SD_obclean_zc = sqlContext.createDataFrame( cleanob_rdd_zc,["loc_lat","loc_long","size", "size_units", "normalized_price", "normalized_size_units", "spec_uuid", "spec_product", "zip_code"])
df_SD_obclean_zc2= df_SD_obclean_zc.filter(df_SD_obclean_zc['zip_code']!='null')
df_SD_obclean_zc2.printSchema()

#clean table
df_SD_obclean_zc2.registerTempTable('clean_observation_table')

# In[21]:

df_us_taxonomy.printSchema()
get_bundle = sqlContext.sql("select distinct bundle from taxonomy ")#get all bundles. This list isn't used because there were some bundle types that were not really bundles
bundle_map = get_bundle.map(lambda p: p.bundle)
bundle_map.persist()
bundle_list = bundle_map.collect()


#add bundle type to the observation_table based on the uuid
observation_with_bundle = sqlContext.sql("select normalized_price, normalized_size_units, spec_uuid, spec_product, zip_code, taxonomy.bundle from clean_observation_table, taxonomy where clean_observation_table.spec_uuid=taxonomy.uuid")

# In[23]:


observation_with_bundle.registerTempTable("ob_with_bundle")
observation_with_bundle_rdd = observation_with_bundle.rdd.coalesce(16) #decrease num of partitions to make collection faster
observation_with_bundle_rdd.persist()
observation_with_bundle_rdd.getNumPartitions()



# In[3]:

from pyspark.sql.types import IntegerType, FloatType
from pyspark.sql.functions import udf

changetofloat = udf(lambda s: float(s), FloatType())

#change normalized_price to float. It was string type before.
ob_with_bundle_float = observation_with_bundle.withColumn("n_price_float", changetofloat(observation_with_bundle.normalized_price))
ob_with_bundle_float.registerTempTable('ob_with_bundle_float')


# In[29]:

df_zc_income_SD.printSchema()

# get average of normalized price for each bundle based on normalized_size_units
ob_with_bundle_float_group = ob_with_bundle_float.groupBy('bundle','normalized_size_units' )
ob_average= ob_with_bundle_float_group.avg('n_price_float').withColumnRenamed("AVG(n_price_float)", "avg_nprice")
ob_average.registerTempTable('ob_average')
# In[26]:


join_average = sqlContext.sql("select normalized_price, ob_with_bundle_float.normalized_size_units, spec_uuid,spec_product, zip_code, ob_with_bundle_float.bundle, n_price_float, avg_nprice  from ob_with_bundle_float, ob_average where ob_with_bundle_float.normalized_size_units= ob_average.normalized_size_units and ob_with_bundle_float.bundle = ob_average.bundle " )
join_average.printSchema()

#get demeaned price per bundle
getdemeaned = udf(lambda s, t: s/t, FloatType())
join_average_demeaned = join_average.withColumn("demeaned_price",getdemeaned(join_average.n_price_float, join_average.avg_nprice) )
join_average_demeaned.registerTempTable('demeaned')



# In[28]:

join_average_demeaned.printSchema()


# In[4]:

################calculations to find weighted AGI per bundle per zip code##############################

#A better and faster way to join these tables would have been to convert the tables to pair RDDs based with the zip code as key and every other value in a tuple. Joining the pair RDDs would take lesser time.
#That would have ensured that each key is in one partition which would make aggregations and joins much faster. But this dataset is small enough for the naive approach to work
AGI_demeaned_price = sqlContext.sql("select AGI, returns, AGI.zipcode, demeaned.bundle, demeaned_price from AGI, demeaned where AGI.zipcode=demeaned.zip_code ")
AGI_demeaned_price.printSchema()
AGI_demeaned_price.registerTempTable("AGI_demeaned_price")
AGIxReturns= AGI_demeaned_price.withColumn("AGIxReturns",AGI_demeaned_price.AGI*AGI_demeaned_price.returns )
AGIxReturns.printSchema()

AGI_demeaned_price_sum_zc = AGIxReturns.groupBy(AGIxReturns.zipcode,AGIxReturns.bundle).agg({"AGIxReturns": "sum", "returns":"sum"}) 
AGI_demeaned_price_sum_zc.printSchema()
renamed_sum_returns = AGI_demeaned_price_sum_zc.withColumnRenamed("SUM(returns)", "sum_returns")
AGI_by_zc_renamed= renamed_sum_returns.withColumnRenamed("SUM(AGIxReturns)", "sum_AGIxReturns")


# In[32]:
# get the weighted AGI for each zip code, bundle
getweight = udf(lambda s,t: s/t, FloatType())

weighted_AGI_zc = AGI_by_zc_renamed.withColumn("weighted_AGI",getweight(AGI_by_zc_renamed.sum_AGIxReturns, AGI_by_zc_renamed.sum_returns) )
weighted_AGI_zc.registerTempTable("weighted_AGI_zc")


# In[33]:
#get food_metric per zip code for every bundle
avg_demeaned_price_zc = AGI_demeaned_price.groupBy('zipcode', 'bundle').avg('demeaned_price').withColumnRenamed("AVG(demeaned_price)", "food_metric")
avg_demeaned_price_zc.registerTempTable("avg_demeaned_price_zc")


# In[34]:

final_table = sqlContext.sql("select weighted_AGI_zc.zipcode, weighted_AGI, food_metric, weighted_AGI_zc.bundle from  weighted_AGI_zc, avg_demeaned_price_zc where weighted_AGI_zc.zipcode=avg_demeaned_price_zc.zipcode and weighted_AGI_zc.bundle = avg_demeaned_price_zc.bundle")
final_table.printSchema()
final_table.registerTempTable('final_table') #final table with schema given below
######
# root
#  |-- zipcode: long (nullable = true)
#  |-- weighted_AGI: float (nullable = true)
#  |-- food_metric: double (nullable = true)
#  |-- bundle: string (nullable = true
########




# In[18]:

#sort final_table by bundle as key. This part is very crucial for fast calculations.

final_table_sorted = final_table.map(lambda p: (p.bundle,(p.zipcode, p.weighted_AGI, p.food_metric)))
final_table_sorted.persist()
final_table_sorted=final_table_sorted.sortByKey().coalesce(16)

# In[23]:

final_table_sorted_df = sqlContext.createDataFrame(final_table_sorted, ['bundle', 'zip_AGI_foodmetric'])


# In[28]:

final_table_sorted_df.registerTempTable('final_table_sorted')
final_table_sorted_df.printSchema()


# In[35]:

#get all bundles 
actual_bundles = sqlContext.sql("select distinct bundle from final_table_sorted")
actual_bundle = actual_bundles.map(lambda p: p.bundle)
actual_bundle.persist()
actual_bundle_list= actual_bundle.collect()
actual_bundle_list #get all bundles in the final_table


# In[52]:

#calculate the pearson correlation coeff for each bundle. Sorting with bundle has key decreased spark job time to only 2 mins from 20 mins per bundle
#used Statistics.corr for the calculation of pearson coeff

from pyspark.mllib.stat import Statistics
bundle_pearson_dict = {} #dictionary to hold the bundle as key and the coeff as value

for bundle_name in actual_bundle_list:
    final_table_by_bundle = sqlContext.sql("select * from final_table_sorted where bundle = \""+bundle_name+"\"")
    food_metric_only= final_table_by_bundle.map(lambda p:  p.zip_AGI_foodmetric[2])
    food_metric_list = food_metric_only.collect()
    weighted_AGI_only= final_table_by_bundle.map(lambda p:  p.zip_AGI_foodmetric[1])
    weighted_AGI_list = weighted_AGI_only.collect()
    if not food_metric_list and not weighted_AGI_list:
        print 'pass'
    else:
        
        x=sc.parallelize(weighted_AGI_list,2)
        y=sc.parallelize(food_metric_list,2)
        
        correlation_coeff =  Statistics.corr(x,y, method="pearson") # -0.128161962745 or is it -0.0965926041863??
        bundle_pearson_dict[bundle_name]= correlation_coeff
    
        
bundle_pearson_dict  #to get all coeff values by bundle

# In[53]:

#Here I have an example scatter plot for bundle_name = 'vegetables' to have an idea of how the plot looks
# x is the AGI for every zip code
# y is the food metric
#an example plot is also available to be viewed in the parent folder

final_table_by_bundle = sqlContext.sql("select * from final_table_sorted where bundle = 'vegetables'")
food_metric_only= final_table_by_bundle.map(lambda p:  p.zip_AGI_foodmetric[2])
food_metric_list = food_metric_only.collect()
weighted_AGI_only= final_table_by_bundle.map(lambda p:  p.zip_AGI_foodmetric[1])
weighted_AGI_list = weighted_AGI_only.collect()
zip_code_list = final_table_by_bundle.map(lambda p:  p.zip_AGI_foodmetric[0]).collect()


# In[2]:

get_ipython().magic(u'matplotlib inline')
import numpy as np; import matplotlib.pyplot as plt
y = food_metric_list # points to plot
x = weighted_AGI_list
labels = zip_code_list # labels of the points
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.scatter(x, y)
plt.ylabel('Food metric')
plt.xlabel('Weight Adjusted Gross Income ($)')
plt.title('Pearson correlation coefficient scatter plot for vegetables (zip code mentioned per point)')
#to label each point according to the zip code
for i, txt in enumerate(labels):
    ax.annotate(txt, (x[i],y[i]))


'''
All the code below has been adapted from http://www.christianpeccei.com/zipmap/. I made an example map plot of Adjusted Gross Income (AGI) on a map of California. More maps like these could potentially be generated which would give both AGI and
food metric for each zip code per bundle. Below is only an example map to just give an idea of other visualizations that could be achieved with the data
'''


def read_ascii_boundary(filestem):
    '''
    Reads polygon data from an ASCII boundary file.
    Returns a dictionary with polygon IDs for keys. The value for each
    key is another dictionary with three keys:
    'name' - the name of the polygon
    'polygon' - list of (longitude, latitude) pairs defining the main
    polygon boundary
    'exclusions' - list of lists of (lon, lat) pairs for any exclusions in
    the main polygon
    '''
    metadata_file = filestem + 'a.dat'
    data_file = filestem + '.dat'
    # Read metadata
    lines = [line.strip().strip('"') for line in open(metadata_file)]
    polygon_ids = lines[::6]
    polygon_names = lines[2::6]
    polygon_data = {}
    for polygon_id, polygon_name in zip(polygon_ids, polygon_names):
        # Initialize entry with name of polygon.
        # In this case the polygon_name will be the 5-digit ZIP code.
        polygon_data[polygon_id] = {'name': polygon_name}
    del polygon_data['0']
    # Read lon and lat.
    f = open(data_file)
    for line in f:
        fields = line.split()
        if len(fields) == 3:
            # Initialize new polygon
            polygon_id = fields[0]
            polygon_data[polygon_id]['polygon'] = []
            polygon_data[polygon_id]['exclusions'] = []
        elif len(fields) == 1:
            # -99999 denotes the start of a new sub-polygon
            if fields[0] == '-99999':
                polygon_data[polygon_id]['exclusions'].append([])
        else:
            # Add lon/lat pair to main polygon or exclusion
            lon = float(fields[0])
            lat = float(fields[1])
            if polygon_data[polygon_id]['exclusions']:
                polygon_data[polygon_id]['exclusions'][-1].append((lon, lat))
            else:
                polygon_data[polygon_id]['polygon'].append((lon, lat))
    return polygon_data



import csv
from pylab import *

# Read in ZIP code boundaries for California
d = read_ascii_boundary('/Users/barshashrestha/Downloads/zipmap/data/zip5/zt06_d00')

# Read in data for AGI by ZIP code in California
f = csv.reader(open('/Users/barshashrestha/Downloads/zipmap/data/veg_AGI_copy.txt', 'rb'))
AGI = {}
# Skip header line
f.next()
# Add data for each ZIP code
for row in f:
    zipcode,AGI_col = row
    AGI[zipcode] = float(AGI_col)
max_AGI = max(AGI.values())

# Create figure and two axes: one to hold the map and one to hold
# the colorbar
figure(figsize=(5, 5), dpi=30)
map_axis = axes([0.0, 0.0, 0.8, 0.9])
cb_axis = axes([0.83, 0.1, 0.03, 0.6])

# Define colormap to color the ZIP codes.
# You can try changing this to cm.Blues or any other colormap
# to get a different effect
cmap = cm.PuRd

# Create the map axis
axes(map_axis)
axis([-125, -114, 32, 42.5])
gca().set_axis_off()

# Loop over the ZIP codes in the boundary file
for polygon_id in d:
    polygon_data = array(d[polygon_id]['polygon'])
    zipcode = d[polygon_id]['name']
    AGI_col = AGI[zipcode] if zipcode in AGI else 0.
    # Define the color for the ZIP code
    fc = cmap(AGI_col / max_AGI)
    # Draw the ZIP code
    patch = Polygon(array(polygon_data), facecolor=fc,
        edgecolor=(.3, .3, .3, 1), linewidth=.2)
    gca().add_patch(patch)
title('AGI per zip code in San Diego')

# Draw colorbar
cb = mpl.colorbar.ColorbarBase(cb_axis, cmap=cmap,
    norm = mpl.colors.Normalize(vmin=0, vmax=max_births))
cb.set_label('AGI')

# Change all fonts to Arial
for o in gcf().findobj(matplotlib.text.Text):
    o.set_fontname('Arial')

# Export figure to bitmap
savefig('/Users/barshashrestha/Documents/SD_AGI.png')
