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
			temp = planet(mass=j['mass'], radius=j['radius'], loanode=j['loanode'], period=j['period'], name=i,
				eccentricity=j['eccentricity'], smaxis=j['smaxis'],argperiapsis=j['argperiapsis'],
				orbital_inclination=j['orbital inclination'],mean_longitude=j['mean longitude'])
			setattr(self,i,temp)

	def initialize(self) -> None:
		return [getattr(self,i) for i in self.rawdata]

	def createnewplanet(self,_mass:int=0,_radius:int=0,_initial_position:np.ndarray=np.array([]),
		_initial_velocity:np.ndarray=np.array([]),_loanode:float=0,_period:float=0,_name:str='Planet9',
		_eccentricity:float=0,_smaxis:int=0,_argperiapsis:float=0,_orbital_inclination:float=0,_mean_longitude:float=0) -> bool:

		"""
		Function to create a new planet, 'Planet9' is used as default name. Returns True on succes
		"""

		if not _mass and _radius:
			raise ValueError('No mass or radius was specified, try again.')
		check1 = _loanode and _eccentricity and _smaxis and _argperiapsis and _orbital_inclination
		check2 = _initial_position and _initial_velocity
		if not check1:
			warnings.warn('Not enough data specified to use this planet in Kepler coordinates.')
		elif not check2:
			warnings.warn('Not enough data specified to use this planet in the ordinary motion equation.')
		elif not check1 and not check2:
			raise ValueError('Not enough data specified to use this planet in simulations.')
		else:
			if check1:
				temp = planet(mass=_mass,radius=_radius,loanode=_loanode,eccentricity=_eccentricity,smaxis=_smaxis,
					argperiapsis=_argperiapsis, orbital_inclination=_orbital_inclination, name=_name, mean_longitude=_mean_longitude)
				setattr(self,_name,temp)
				return True
			elif check2:
				temp = planet(mass=_mass,radius=_radius,initial_velocity=_initial_velocity,initial_position=_initial_position)
				setattr(self,_name,temp)
				return True

	def __str__(self):
		return "Please don't print me, ask for my rawdata instead. I'll help you a bit though. \n" + str(self.rawdata)