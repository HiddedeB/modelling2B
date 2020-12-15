from PlanetClass import planet
import warnings, numpy as np, json
from numba import types, typed, typeof
from numba.experimental import jitclass

# Deprecated
'''
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

	def createnewplanet(self,mass:float=0,radius:int=0,initial_position:np.ndarray=np.array([]),
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
			self.rawdata[name] = {"mass":float(mass),"radius":float(radius),"loanode":float(loanode),"eccentricity":float(eccentricity),
				"smaxis":float(smaxis), "argperiapsis":float(argperiapsis), "orbital inclination":float(orbital_inclination),
				"mean longitude":float(mean_longitude),"period":float(period)}
			j = self.rawdata[name]
			temp = planet(mass=j['mass'], radius=j['radius'], loanode=j['loanode'], period=j['period'], name=name,
				eccentricity=j['eccentricity'], smaxis=j['smaxis'],argperiapsis=j['argperiapsis'],
				orbital_inclination=j['orbital inclination'],mean_longitude=j['mean longitude'])
			setattr(self,name,temp)
			return True
		elif check2:
			self.rawdata[name] = {"mass":float(mass),"radius":float(radius),"initial velocity":initial_velocity,
			"initial position":initial_position}
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

	def set_kuyperbelt(self,total_mass:float,r_res:int,range:list,hom_mode:bool=False):
		"""Creates an array of planet objects for the euler lagrange method
		hom_mode False divides their mass evenly over orbits, True divides it homogeneously in
		the radial direction."""
		radii = np.linspace(range[0],range[1],r_res)
		if hom_mode:
			m1 = total_mass*range[0]/np.sum(radii)
			weights = radii/range[0]
			masses = weights*m1
		else:
			masses = np.repeat(total_mass/r_res,r_res)
		_orbital_inclination = 1.8 #https://arxiv.org/abs/1704.02444 	I
		_loanode = 77 #https://arxiv.org/abs/1704.02444					big Omega
		_eccentricity = 0.1 #DES/SSBN07 classification					e
		roids = np.array([])
		for i in range(len(radii)):
			asteroid = planet(mass=masses[i],radius=0,eccentricity=_eccentricity,loanode=_loanode,orbital_inclination=_orbital_inclination,
				smaxis=radii[i])
			roids = np.append(roids,asteroid)
		setattr(self,"asteroids_array",roids)
'''
#specify specs
dict_kv_ty = (types.unicode_type,types.float64)
dicttype = types.DictType(*dict_kv_ty)
kv_ty2 = (types.unicode_type,types.float64[:])
dicttype2 = types.DictType(*kv_ty2)
with open('data.json') as file:
	dicts = [i for i in json.load(file)]

if not "planet9" in dicts:
	dicts.append("planet9")
spec = [(i,dicttype) for i in dicts]
spec += [('asteroids',types.ListType(dicttype))]
d = typed.Dict.empty(key_type=types.unicode_type,value_type=dicttype)
spec += [('rawdata',typeof(d))]
spec += [('asteroid_attributes',dicttype2)]

def create_pdh(filename1,filename2=False):
	with open(filename1) as file:
		data = json.load(file)
	tojitclass = typed.Dict()
	for i in data:
		temp = typed.Dict()
		for j in data[i]:
			temp[j]=data[i][j]
		tojitclass[i]=temp
	if filename2:
		with open(filename2) as file:
			data = json.load(file)
		for i in data:
			temp = typed.Dict()
			for j in data[i]:
				temp[j]=float(data[i][j])
			tojitclass[i]=temp
	return JitPDH(tojitclass)

@jitclass(spec)
class JitPDH:
	def __init__(self,rawdata):

		self.rawdata = rawdata
		self.sun = typed.Dict.empty(*dict_kv_ty)
		self.mercury = typed.Dict.empty(*dict_kv_ty)
		self.venus = typed.Dict.empty(*dict_kv_ty)
		self.earth = typed.Dict.empty(*dict_kv_ty)
		self.mars = typed.Dict.empty(*dict_kv_ty)
		self.jupiter = typed.Dict.empty(*dict_kv_ty)
		self.saturn = typed.Dict.empty(*dict_kv_ty)
		self.uranus = typed.Dict.empty(*dict_kv_ty)
		self.neptune = typed.Dict.empty(*dict_kv_ty)
		self.planet9 = typed.Dict.empty(*dict_kv_ty)
		self.asteroids = typed.List.empty_list(dicttype)
		self.asteroid_attributes = typed.Dict.empty(*kv_ty2)
		self.asteroid_attributes['mass']=np.zeros(0,dtype=np.float64)
		self.asteroid_attributes['eccentricity']=np.zeros(0,dtype=np.float64)
		self.asteroid_attributes['loanode']=np.zeros(0,dtype=np.float64)
		self.asteroid_attributes['smaxis']=np.zeros(0,dtype=np.float64)
		self.asteroid_attributes['orbital inclination']=np.zeros(0,dtype=np.float64)
		self.asteroid_attributes['argperiapsis']=np.zeros(0,dtype=np.float64)

		for i in rawdata:
			temp = rawdata[i]
			if i == "sun":
				self.sun = temp
			elif i == "mercury":
				self.mercury = temp
			elif i== "venus":
				self.venus = temp
			elif i == "earth":
				self.earth = temp
			elif i == "mars":
				self.mars = temp
			elif i == "jupiter":
				self.jupiter = temp
			elif i == "saturn":
				self.saturn = temp
			elif i == "uranus":
				self.uranus = temp
			elif i == "neptune":
				self.neptune = temp
			elif i == "planet9":
				self.planet9 = temp

	def createnewplanet(self,mass=0,radius=0,loanode=0,
		eccentricity=0,smaxis=0,argperiapsis=0,orbital_inclination=0,mean_longitude=0):

		"""
		Function to create a new planet with name 'Planet9'. Returns True on succes.
		Keyword arguments can only be specified as non-keyword because of JIT.
		Use createnewplanet(1,2) for a planet with mass 1 and radius 2.
		"""
		self.planet9['mass']=mass
		self.planet9['radius']=radius
		self.planet9['loanode']=loanode
		self.planet9['eccentricity']=eccentricity
		self.planet9['smaxis']=smaxis
		self.planet9['argperiapsis']=argperiapsis
		self.planet9['orbital inclination']=orbital_inclination
		self.planet9['mean longitude']=mean_longitude

	def __str__(self):
		# Dit werkt niet for some reason
		return "Please don't print me, ask for my rawdata instead. I'll help you a bit though. \n" + str(self.rawdata)

	def add_etnos(self):
		for i in self.rawdata:
			if i not in ['sun','mercury','venus','earth','mars','jupiter','uranus','neptune','saturn','planet9']:
				self.asteroids.append(self.rawdata[i])
				temp = self.asteroid_attributes
				self.asteroid_attributes['mass'] = np.append(temp['mass'],self.rawdata[i]['mass'])
				self.asteroid_attributes['smaxis'] = np.append(temp['smaxis'],self.rawdata[i]['smaxis'])
				self.asteroid_attributes['eccentricity'] = np.append(temp['eccentricity'],self.rawdata[i]['eccentricity'])
				self.asteroid_attributes['loanode'] = np.append(temp['loanode'],self.rawdata[i]['loanode'])
				self.asteroid_attributes['orbital inclination'] = np.append(temp['orbital inclination'],self.rawdata[i]['orbital inclination'])
				self.asteroid_attributes['argperiapsis'] = np.append(temp['argperiapsis'],self.rawdata[i]['argperiapsis'])

	def add_kuiperbelt(self,total_mass,r_res,range_min,range_max,hom_mode=False):
		"""Creates an array of planet objects for the euler lagrange method
		hom_mode False divides their mass evenly over orbits, True divides it homogeneously in
		the radial direction."""
		radii = np.linspace(range_min,range_max,r_res)
		if hom_mode:
			m1 = total_mass*range_min/np.sum(radii)
			weights = radii/range_max
			masses = weights*m1
		else:
			masses = np.repeat(np.array([total_mass/r_res],dtype=np.float64),r_res)
		_orbital_inclination = 1.8 #https://arxiv.org/abs/1704.02444
		_loanode = 77 #https://arxiv.org/abs/1704.02444
		_eccentricity = 0.1 #DES/SSBN07 classification
		_argperiapsis = 270 #Nergens op gebaseerd
		self.asteroid_attributes['mass'] = np.append(self.asteroid_attributes['mass'],masses)
		self.asteroid_attributes['smaxis'] = np.append(self.asteroid_attributes['smaxis'],radii)
		self.asteroid_attributes['eccentricity'] = np.append(self.asteroid_attributes['eccentricity'],np.repeat(np.array([_eccentricity],dtype=np.float64),r_res))
		self.asteroid_attributes['loanode'] = np.append(self.asteroid_attributes['loanode'],np.repeat(np.array([_loanode],dtype=np.float64),r_res))
		self.asteroid_attributes['orbital inclination'] = np.append(self.asteroid_attributes['orbital inclination'],np.repeat(np.array([_orbital_inclination],dtype=np.float64),r_res))
		self.asteroid_attributes['argperiapsis'] = np.append(self.asteroid_attributes['argperiapsis'],np.repeat(np.array([_argperiapsis],dtype=np.float64),r_res))
		for i in range(len(radii)):
			asteroid = typed.Dict()
			asteroid['mass']=masses[i]
			asteroid['eccentricity']=_eccentricity
			asteroid['loanode']=_loanode
			asteroid['argperiapsis']=_argperiapsis
			asteroid['orbital inclination']=_orbital_inclination
			asteroid['smaxis']=radii[i]
			self.asteroids.append(asteroid)

	def reset_asteroids(self):
		self.asteroids = typed.List.empty_list(dicttype)
		self.asteroid_attributes = typed.Dict.empty(*kv_ty2)
		self.asteroid_attributes['mass']=np.zeros(0,dtype=np.float64)
		self.asteroid_attributes['eccentricity']=np.zeros(0,dtype=np.float64)
		self.asteroid_attributes['loanode']=np.zeros(0,dtype=np.float64)
		self.asteroid_attributes['smaxis']=np.zeros(0,dtype=np.float64)
		self.asteroid_attributes['orbital inclination']=np.zeros(0,dtype=np.float64)
		self.asteroid_attributes['argperiapsis']=np.zeros(0,dtype=np.float64)

if __name__=='__main__':
	pdh = create_pdh('data.json','etnos.json')