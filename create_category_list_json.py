import json
categories ={
		'Mallard Male':{},
		'Mallard Female':{},
		'Ring-necked duck Male':{},
		'Ring-necked duck Female':{},
		'Green-winged teal':{},
		'Other':{
			'Dabbler':{
				'Shoveler':['Male','Female','Unknown'],
				'Pintail':['Male','Female','Unknown'],
				'Gadwall':[],
				'American Widgeon':['Male','Female','Unknown'],
				'Wood Duck':['Male','Female','Unknown'],
				'Other Dabbler (fill in species)':[]
				},
			'Merganser':{
				'Hooded Merganser':['Male','Female','Unknown'],
				'Common Merganser':['Male','Female','Unknown'],
				},
			'Diver':{
				'Scaup':['Male','Female','Unknown'],
				'Canvasback':['Male','Female','Unknown'],
				'Redhead':['Male','Female','Unknown'],
				'Other Diver (fill in species)':[]
				},
			},
		'Goose':{
			'White-fronted Goose':{},
			'Canada Goose':{},
			'Snow/Ross Goose':{},
			'Snow/Ross Goose (blue)':{},
			'Swan':{},
			},
		'REDUnknown':{},
		'REDMultiple Birds':{},
		'REDpartial bird':{},#Box not around whole bird/not centered on bird (partial bird)
		'REDNot target species':{},
		'REDNot a bird':{},
		}

data = {"class_list":categories,
	"GUIResolution": [1400, 800],
	"RelativeLayoutImageView": [0.6666666666666666, 1], 
	"RelativeLayoutBirdView": [0.16666666666666666, 0.25]}

if __name__=='__main__':
	data = {"class_list":categories,
	"GUIResolution": [1200, 800],
	"RelativeLayoutImageView": [0.6666666666666666, 1], 
	"RelativeLayoutBirdView": [0.16666666666666666, 0.25]}
	with open('./new_category.json', 'w') as f:
		json.dump(data,f)