import numpy
import pandas
import pprint
from matplotlib import pyplot

class DataSetName:
    InternationalAirlinePassengers = 1
    GlobalLandTemperaturesByCountry = 2
    IstanbulStockExchangeIndex = 3
    AirQuality = 4
    IBM = 5
    HouseholdPowerConsumption = 6
    DomoticHouse = 7
    Sunspots = 8

class DataSetCollection():
    def __GlobalLandTemperaturesByCountry(self):
        dateparse = lambda dates: pandas.datetime.strptime(dates, '%Y-%m-%d')
        dta = pandas.read_csv('GlobalLandTemperaturesByCountry.csv', sep=',',
                              usecols=['dt', 'AverageTemperature', 'Country'],
                              parse_dates=['dt'], index_col='dt', engine='python', date_parser=dateparse)
        dta = dta[dta.Country == 'Bangladesh']
        dta.interpolate(inplace=True)
        dataset = dta.AverageTemperature.values[:, numpy.newaxis].astype('float32')
        return dataset

    def __InternationalAirlinePassengers(self):
        dataframe = pandas.read_csv('international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
        dataset = dataframe.values
        dataset = dataset.astype('float32')
        return dataset

    def __IstanbulStockExchangeIndex(self):
        dataframe = pandas.read_csv('istanbul_stock__exchange_index.csv', usecols=[2], engine='python', skipfooter=3)
        dataframe = dataframe.abs() * 100
        dataset = dataframe.values
        dataset = dataset.astype('float32')
        return dataset

    def __AirQuality(self):
        dataframe = pandas.read_csv('AirQualityUCI.csv', sep=',', usecols=[2], engine='python', skipfooter=3)
        dataframe.interpolate(inplace=True)
        dataset = dataframe.values
        dataset = dataset.astype('float32')
        return dataset

    def __IBM(self):
        dataframe = pandas.read_csv('ibm.csv', sep=',', usecols=[4], engine='python', skipfooter=3)
        dataframe.interpolate(inplace=True)
        dataset = dataframe.values
        dataset = dataset.astype('float32')
        return dataset

    def __HouseholdPowerConsumption(self):
        dataframe = pandas.read_csv('household_power_consumption.csv', sep=';', usecols=[2], engine='python',
                                    skipfooter=3)
        dataframe = dataframe.replace({'?': numpy.nan})
        dataframe.dropna(inplace=True)
        dataset = dataframe.tail(60*24).values
        dataset = dataset.astype('float32')
        return dataframe

    def __DomoticHouse(self):
        dataframe = pandas.read_csv('NEW-DATA-1.T15.csv', sep=',', usecols=[21], engine='python',
                                    skipfooter=3)
        dataframe.dropna(inplace=True)
        dataset = dataframe.values
        dataset = dataset.astype('float32')
        return dataframe

    def __Sunspots(self):
        dataframe = pandas.read_csv('Sunspots.csv', sep=',', usecols=[1], engine='python',
                                    skipfooter=3)
        dataframe.dropna(inplace=True)
        dataset = dataframe.values
        dataset = dataset.astype('float32')
        return dataframe

    def load_dataset(self, datasetName):
        if (datasetName == DataSetName.GlobalLandTemperaturesByCountry):
            return self.__GlobalLandTemperaturesByCountry()
        if (datasetName == DataSetName.InternationalAirlinePassengers):
            return self.__InternationalAirlinePassengers()
        if (datasetName == DataSetName.IstanbulStockExchangeIndex):
            return self.__IstanbulStockExchangeIndex()
        if (datasetName == DataSetName.AirQuality):
            return self.__AirQuality()
        if (datasetName == DataSetName.IBM):
            return self.__IBM()
        if (datasetName == DataSetName.HouseholdPowerConsumption):
            return self.__HouseholdPowerConsumption()
        if (datasetName == DataSetName.DomoticHouse):
            return self.__DomoticHouse()
        if (datasetName == DataSetName.Sunspots):
            return self.__Sunspots()


#d = DataSetCollection()
# ds = d.load_dataset(DataSetName.GlobalLandTemperaturesByCountry);
# print(ds.shape)
# pprint.pprint(ds)
# ds = d.load_dataset(DataSetName.InternationalAirlinePassengers);
# print(ds.shape)
# pprint.pprint(ds)
#ds = d.load_dataset(DataSetName.IstanbulStockExchangeIndex);
#print(ds.shape)
#pprint.pprint(ds)

# ds = d.load_dataset(DataSetName.Sunspots);
# print(ds.shape)
# pprint.pprint(ds)

#pyplot.plot(ds )
#pyplot.show()
