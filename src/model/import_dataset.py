import encoding_decoding as ed #

class ImportDataset():
    def __init__(self, file_path):
        #super.__init__()
        self.file_path = file_path
        edc = ed.EncodeDecode(self.file_path) #
    # def __iter__(self):
    def importData(self):
        with open(self.file_path, 'r') as f:
            cube_data = json.load(f)
            #cube_data = json.read(f)
            cube=cube_data["solution"]
            # set default values: 
            cst, mv, amvst = None, None, None
            ##result = self.get_nested_value(cube)
            #if result:
            #if result not empty:
                #cst, mv, amvst = result
            #else:
                # Handle the case where no data was found
                #continue
            for result in self.get_nested_value(cube):
                cst, mv, amvst = result
                # ... rest of your processing and yield ...
                # Replace your current result check with this:
                ##result = next(self.get_nested_value(cube), None)
                if result is not None:
                    cst, mv, amvst = result
                else:
                    # Handle empty case
                    print(f"No data found for cube")
                    continue
                print(f"result={result}")
                #cst, mv, amvst = self.get_nested_value(cube)
                # cst = torch.tensor(encode(cst), dtype=torch.long)
                cst = torch.tensor(edc.encode(cst), dtype=torch.long)
                # mv = torch.tensor(encode(mv), dtype=torch.long)
                mv = torch.tensor(edc.encode(mv), dtype=torch.long)
                # amvst = torch.tensor(encode(afmvst), dtype=torch.long)
                amvst = torch.tensor(edc.encode(afmvst), dtype=torch.long)
                yield torch.tensor(cst), torch.tensor(mv), torch.tensor(amvst)
    def get_nested_value(self,data):
        """##result = self.get_nested_value(cube)
        Recursively searches for a target_key in a nested dictionary.
        """
        mv=[]
        cst = {}
        amst={}
        # If the current element is a dictionary, look inside
        if isinstance(data, dict):
            for key, value in data.items():
                if key == "state":
                    #yield value
                    cst=value
                elif len(key) == 3:
                    if len(value)==20 :
                        #yield value
                        mv=key
                        amvst=value
                    elif len(value)== (16, 19):
                        #yield value
                        mv=key
                        amvst=value["state"]
                if cst and mv and amvst and key!="state":
                    yield cst, mv, amvst
                if isinstance(value, dict) and len(value)==(16, 19) :
                    # If the value is another dictionary, dive deeper (recursion)
                    yield from get_nested_value(value)
                
