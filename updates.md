<h1 id="mar-update">March 2023 Update:</h1>
<p style="font-family:verdana">We have now upgraded `FB Prophet` to the latest version which is simply called `prophet`. <br>

<h1 id="aug-update">Aug 2022 Update:</h1>
<p style="font-family:verdana">You can now add FB Prophet arguments directly into Auto_TimeSeries using the kwargs argument. See example below:

![fb-prophet](images/add_fb_prophet.png)

<h1 id="jan-update">Jan 2022 Update:</h1>
New since version 0.0.35: You can now load your file into a Dask dataframe automatically. Just provide the name of your file and if it is too large to fit into a pandas dataframe, Auto_TS will automatically detect and load it into a Dask dataframe.

