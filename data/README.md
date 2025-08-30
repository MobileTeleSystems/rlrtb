## Description
Preprocessed iPinYou dataset sample for advertising campaign 1458. The sample is supplemented with several values for each timestamp to simulate competition between agents. 

In total, the dataset contains 1446090 rows for unique 289218 timestamps.

## Data format

- **timestamp** (numpy.int64): format yyyyMMddHHmmssSSS
- **weekday** (numpy.int64): day of week 0-6
- **hour** (numpy.int64): hour of day
- **bidid** (str): bid identifier
- **logtype** (numpy.int64): impression flag
- **ipinyouid** (str): internal user id set by iPinYou
- **useragent** (str): user's device, browser and opertaion system
- **region** (numpy.int64): region id
- **city** (numpy.int64): city id
- **adexchange** (numpy.int64): ad exchange id
- **IP** (str): user's IP adress
- **domain** (str): domain name
- **url** (str): URL of the hosting webpage of the ad slot
- **urlid** (numpy.float64): URL, masked by ad exchanges
- **slotid** (str): slot id
- **slotwidth** (numpy.int64): slot width
- **slotheight** (numpy.int64): slot height
- **slotvisibility** (numpy.int64): describes if the ad slot is above the fold ('FirstView') or not ('SecondView'), or unknown ('Na')
- **slotformat** (numpy.int64): possible values include 'Fixed' (fixed size and position), 'Pop' (the pop-up window), 'Background', 'Float', and 'Na' which presents unknown cases.
- **creative** (str): creative id
- **slotprice** (numpy.int64): bid floor
- **bidprice** (numpy.int64): bid price
- **payprice** (numpy.int64): market price
- **pCTR** (numpy.float64): predicted CTR
- **keypage** (str): key page URL
- **advertiser** (numpy.int64): advertiser id
- **usertag** (str): parsed into user_os and user_browser 
- **click** (numpy.int64): click or not click
- **nclick** (numpy.int64): number of clicks
- **nconversation** (numpy.int64): number of conversions