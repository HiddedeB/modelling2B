from PlanetClass import planet
import warnings, numpy as np, json

# #specify specs
# with open('data.json') as file:
# 	planets_in_data = [i for i in json.load(file)]
# spec = [('rawdata',dict)]
# spec.append([(i,planet) for i in planets_in_data])

# @jitclass gaat alleen werken als we alleen planet9 toe kunnen voegen, of iig iets moeten hardcoden.
class PlanetaryDataHandler:
	def __init__(self):
		with open('data.json') as file:
			self.rawdata = json.load(file)
		for i in self.rawdata:
			j = self.rawdata[i]
			if i != "sun":
				temp = planet(mass=j['mass'], radius=j['radius'], loanode=j['loanode'], period=j['period'], name=i,
					eccentricity=j['eccentricity'], smaxis=j['smaxis'],argperiapsis=j['argperiapsis'],
					orbital_inclination=j['orbital inclination'],mean_longitude=j['mean longitude'])
			else:
				temp = planet(mass=j['mass'],radius=j['radius'], name=i)
			setattr(self,i,temp)

	def get_planets(self) -> np.ndarray:
		return np.array([getattr(self,i) for i in self.rawdata])

	def createnewplanet(self,mass:int=0,radius:int=0,initial_position:np.ndarray=np.array([]),
		initial_velocity:np.ndarray=np.array([]),loanode:float=0,period:float=0,name:str='Planet9',
		eccentricity:float=0,smaxis:int=0,argperiapsis:float=0,orbital_inclination:float=0,mean_longitude:float=0) -> bool:

		"""
		Function to create a new planet, 'Planet9' is used as default name. Returns True on succes
		"""

		if not mass or not radius:
			raise ValueError('No mass or radius was specified, try again.')
		check1 = smaxis>0
		check2 = initial_position==np.array([])
		if not check1 and not check2:
			raise ValueError('Not enough data specified to use this planet in simulations.')
		elif not check2:
			warnings.warn('Not enough data specified to use this planet in the ordinary motion equation.')
		elif not check1:
			warnings.warn('Not enough data specified to use this planet in Kepler coordinates.')
		if check1:
			self.rawdata[name] = {"mass":mass,"radius":radius,"loanode":loanode,"eccentricity":eccentricity,"smaxis":smaxis,
				"argperiapsis":argperiapsis, "orbital inclination":orbital_inclination,"mean longitude":mean_longitude,"period":period}
			j = self.rawdata[name]
			temp = planet(mass=j['mass'], radius=j['radius'], loanode=j['loanode'], period=j['period'], name=name,
				eccentricity=j['eccentricity'], smaxis=j['smaxis'],argperiapsis=j['argperiapsis'],
				orbital_inclination=j['orbital inclination'],mean_longitude=j['mean longitude'])
			setattr(self,name,temp)
			return True
		elif check2:
			self.rawdata[name] = {"mass":mass,"radius":radius,"initial velocity":initial_velocity,"initial position":initial_position}
			j = self.rawdata[name]
			temp = planet(mass=j['mass'],radius=j['radius'],initial_velocity=j['initial velocity'],initial_position=j['initial position'])
			setattr(self,name,temp)
			return True

	def __str__(self):
		return "Please don't print me, ask for my rawdata instead. I'll help you a bit though. \n" + str(self.rawdata)

	def save(self):
		"""Save using console, provides a walkthrough. Filename can be given with or without '.json'."""
		print([i for i in self.rawdata])
		selection = input("Which planets would you like to save? Type the names from the list given above separated with a space, leave empty for all.\n")
		try:
			if selection == '':
				out = self.rawdata
			else:
				out = {i:self.rawdata[i] for i in selection.split()}
			w = True
		except Exception as e:
			print(e)
			w = False
		if w:
			fname = input('Please enter the desired filename: ')
			if not fname[-5:] == '.json':
				fname = fname + '.json'
			with open(fname,'w') as file:
				json.dump(out,file,indent='\t')

	def save2(self,filename:str,selection:list=[]):
		"""Save using script, give a selection like ['mercury','earth'] and a filename with or without '.json'."""
		if selection == []:
			out = self.rawdata
		else:
			out = {i:self.rawdata[i] for i in selection}
		if not filename[-5:] == '.json':
			filename = filename + '.json'
		with open(filename,'w') as file:
			json.dump(out,file,indent='\t')
