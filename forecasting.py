from neuralprophet import NeuralProphet
import warnings
import pickle
import pandas as pd
import warnings
import shutil
import numpy as np
from datetime import timedelta
from pathlib import Path


BASE_DIR = Path(__file__).parent.parent


class Forecasting:


    def __init__(self, output_dir):
        self.output_dir = output_dir

    def __fit_model_pkl(self, df, combo):

        """
        This function adjust jobsite availabilities by taking into account safety stock
        
        :param jobsite_avail_df: table of jobsite availabilities
        :type nodes_df: pandas.DataFrame
        :returns: table of jobsite availabilities accounting for safety stock
        :rtype: pandas.DataFrame
        """

        df.columns = ['ds', 'y']

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # fit model
            model = NeuralProphet(epochs=60, batch_size=40)  
            model.fit(df, freq="D"); 

            # ensure temp directory exists
            temp = Path(self.output_dir, "forecasts", "models_TEMP")
            temp.mkdir(parents=True, exist_ok=True)

            # generate file path
            filename = "NP_" + str(combo) + ".pkl"
            pickle_path = Path(temp, filename)
            
            # create .pkl file
            with pickle_path.open("wb") as f:
                pickle.dump(model, f)


    def __move_files(self, source_dir, dest_dir):

        """
        This function adjust jobsite availabilities by taking into account safety stock
        
        :param jobsite_avail_df: table of jobsite availabilities
        :type nodes_df: pandas.DataFrame
        :returns: table of jobsite availabilities accounting for safety stock
        :rtype: pandas.DataFrame
        """
        
        # iterate through directory non-recursively
        for file_path in Path(source_dir).iterdir():
            
            # check if file, then move file via rename()
            if file_path.is_file():
                Path(file_path).rename(Path(dest_dir, file_path.name))


    def __avoid_duplicates(self, directory, filename):

        """
        This function adjust jobsite availabilities by taking into account safety stock
        
        :param jobsite_avail_df: table of jobsite availabilities
        :type nodes_df: pandas.DataFrame
        :returns: table of jobsite availabilities accounting for safety stock
        :rtype: pandas.DataFrame
        """
        
        for file_path in Path(directory).iterdir():

            if filename == file_path.name:
                return True
        
        return False
        

    def create_models(self, input_df):

        """
        This function adjust jobsite availabilities by taking into account safety stock
        
        :param jobsite_avail_df: table of jobsite availabilities
        :type nodes_df: pandas.DataFrame
        :returns: table of jobsite availabilities accounting for safety stock
        :rtype: pandas.DataFrame
        """
        
        # ensure temp directory exists
        temp_dir = Path(self.output_dir, "forecasts", "models_TEMP")
        temp_dir.mkdir(parents=True, exist_ok=True)

        # ensure main directory exists
        main_dir = Path(self.output_dir, "forecasts", "models_MAIN")
        main_dir.mkdir(parents=True, exist_ok=True)

        # start count of models produced by alg
        num_models = 0

        # format dates in input dataframes
        input_df['Date'] = pd.to_datetime(input_df["Date"])
        latest_date = input_df['Date'].max()

        # get list of unique cluster-item combos
        combos = input_df['Combo'].unique()

        # initialize repeats df for helping store repeats data
        repeats_df=pd.DataFrame(columns=['Cluster','ItemCode','Item','Date','Qty','Combo'])

        for combo in combos:
            
            # get last 90 days dataframe, filtered by combo
            filtered_df = input_df[input_df['Combo'] == combo]
            last_90_days_df = filtered_df[filtered_df['Date'] > (latest_date-timedelta(days=90))]

            # if empty, move on
            if last_90_days_df.empty or (last_90_days_df['Qty'].sum()==0):
                pass
                # check if 
            
            # else if all values the same, add to repeats
            elif last_90_days_df['Qty'].mean()==last_90_days_df["Qty"].iloc[-1]:
                temp = last_90_days_df[last_90_days_df['Combo'] == combo].iloc[-1,:]
                temp_df = pd.DataFrame(data=[[temp['Cluster'], temp['ItemCode'], temp['Item'],
                                            temp['Date'], temp['Qty'], temp['Combo']]],
                                            columns=['Cluster','ItemCode','Item','Date','Qty','Combo'])
                repeats_df = pd.concat([repeats_df, temp_df], axis=0, ignore_index = True)
            
            else:
                # every thousand models, move temp_dir files into main_dir
                if num_models % 1000 == 0:
                    self.__move_files(temp_dir, main_dir)

                # define name of .pkl model
                filename = "NP_" + str(combo) + ".pkl"

                # check for duplicates
                is_duplicate = self.__avoid_duplicates(temp_dir, filename) or self.__avoid_duplicates(main_dir, filename)
                
                # if no duplicates, fit model
                if not is_duplicate:
                    try:
                        self.__fit_model_pkl(filtered_df[['Date','Qty']], combo)
                        num_models += 1
                    except:
                        pass
        
        # transfer remaining files, remove temp_dir
        self.__move_files(temp_dir, main_dir)
        shutil.rmtree(temp_dir)

        return repeats_df


    def extract_model_predictions(self, all_combos_df, repeats_df):

        """
        This function adjust jobsite availabilities by taking into account safety stock
        
        :param jobsite_avail_df: table of jobsite availabilities
        :type nodes_df: pandas.DataFrame
        :returns: table of jobsite availabilities accounting for safety stock
        :rtype: pandas.DataFrame
        """

        # ensure Date col is df
        all_combos_df['Date'] = pd.to_datetime(all_combos_df["Date"])
        
        # most recent date in dataframe
        latest_date = all_combos_df['Date'].max()

        # get list of unique cluster-item combos
        combos = all_combos_df['Combo'].unique()

        # set min and max value for the forecast 
        # (there should not be over 1000000 of the same item at a branch)
        floor_min=0     
        cap_max=1000000

        # initialize dataframe with final forecasts
        final_forecasts_df = pd.DataFrame(columns=['ds','yhat1','item'])    

        # check if there is saved .pkl model for each combo
        for combo in combos:

            # get predcited combo path
            combo_path = Path(self.output_dir, "forecasts", "models_MAIN", "NP_" + str(combo) + ".pkl")

            # check if predicted combo path is file
            if combo_path.is_file():

                # extract combo model
                with combo_path.open("rb") as f:
                    combo_model = pickle.load(f)

                # most recent day - 2 (already add extra 2 days and make dataframe starts from last date)
                range = pd.date_range(latest_date-timedelta(days=2), periods=2, freq='D') 
                
                # format dataframe for forecasting
                df = pd.DataFrame({'Date': range})
                df['Qty'] = pd.Series(dtype='int')
                df.columns = ['ds', 'y']
                future1 = combo_model.make_future_dataframe(df, periods=60)
                
                # forecast using model
                forecast = combo_model.predict(future1)
                
                # format forecast dataframe
                forecast1 = forecast[['ds','yhat1']]
                forecast1['yhat1'] = np.maximum(floor_min, np.minimum(cap_max, forecast1['yhat1'].values))
                forecast1['yhat1'] = forecast1['yhat1'].astype(int)
                forecast1['item'] = combo

                # merge into larger dataframe
                final_forecasts_df = pd.concat([final_forecasts_df, forecast1], ignore_index = True) # the 1st time this code runs it will fail due to final_forecast not being defined so then it will pass to the all_forecasts argument which is next

        # creating "forecasts" for the items with repeat quantities
        for _, row in repeats_df.iterrows():

            item = row["Combo"]
            qty = row["Qty"]
            out_df = pd.DataFrame({'ds': pd.date_range(latest_date+timedelta(days=1), periods=60, freq='D'),
                                   'yhat1': qty,
                                   'item': item,
                                  })
            
            final_forecasts_df = pd.concat([final_forecasts_df, out_df], ignore_index = True)

        return final_forecasts_df


    def get_future_state_df(self, final_forecasts_df, current_date, num_future_elapsed_days, clusters, items):

        """
        This function adjust jobsite availabilities by taking into account safety stock
        
        :param jobsite_avail_df: table of jobsite availabilities
        :type nodes_df: pandas.DataFrame
        :returns: table of jobsite availabilities accounting for safety stock
        :rtype: pandas.DataFrame
        """

        final_forecasts_df["ds"] = final_forecasts_df["ds"].apply(pd.to_datetime)
        future_date = pd.to_datetime(current_date + timedelta(days = num_future_elapsed_days))
        future_state_df = final_forecasts_df[final_forecasts_df["ds"] == future_date]

        combos = list(future_state_df["item"].unique())
        
        for cluster in clusters:
            for item in items:
                key = str(cluster) + "__" + str(item)

                if key not in combos:
                    temp_dict = {
                        "ds": [future_date],
                        "yhat1": [0],
                        "item": [key]
                    }
                    temp = pd.DataFrame(data=temp_dict)
                    future_state_df = pd.concat([future_state_df, temp])

        future_state_df.sort_values(by=["item"], inplace=True)

        return future_state_df
