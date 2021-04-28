`JigsawLoader.py` defines different types of `dataset`:
  * JigsawNewDataset(utils.data.Dataset) #for train-datasets
  * JigsawTestNewDataset(JigsawNewDataset) #for val/test-datasets(esp for batch2&3 and Downsampled month data)
  ～～* JigsawTestNewMonthDataset(JigsawNewDataset) #for val/test-dataset(esp for different imbalance ratios' month data)～～
 
 `data_helper.py` defines different types of `dataloader` and `transformer`:
  * get_train_dataloader(args): -> train_loader, val_loader
  * get_val_dataloader(args): -> test_loader
  * get_tgt_dataloader(args): -> loader #Load whole domain dataset(combines train&val&test)
  * get_train_transformers(args): -> transforms(img), transforms(label)
  * get_val_transformer(args): -> transforms
  * get_nex_val_transformer(args): -> transforms #different mean and std norm

# Dataset Path
* `batch2, batch3, Jul/Augu/Sep...(downsampled)` need stored in `data\kfold`
  ```
  #If not exists 'data/kfold':
  mkdir data/kfold
  #Then store all your data folders under this path
  ```
  <img src="https://user-images.githubusercontent.com/39040787/115491263-3a2ec280-a292-11eb-9915-14796de94acb.png" width="200" height="400"/><br/>
  
 # data samples' `path2label.txt` stored in `correct_txt_lists`
 
