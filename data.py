from PlanetClass import planet
import warnings, numpy, json

class PlanetaryDataHandler:
	def __init__(self):
		with open('data.json') as file:
			self.rawdata = json.load(file)
		for i in self.rawdata:
			j = self.rawdata[i]
			temp = planet(mass=j['mass'], radius=j['radius'], loanode=j['loanode'], period=j['period'], name=i,
				eccentricity=j['eccentricity'], smaxis=j['smaxis'],
				argperiapsis=j['argperiapsis'], orbital_inclination=j['orbital inclination'])
			setattr(self,i,temp)

	def initialize(self) -> None:
		return [getattr(self,i) for i in self.rawdata]

	def createnewplanet(self,_mass:int=0,_radius:int=0,_initial_position:np.ndarray=np.array([]),
		_initial_velocity:np.ndarray=np.array([]),_loanode:float=0,_period:float=0,_name:str='Planet9',
		_eccentricity:float=0,_smaxis:int=0,_argperiapsis:float=0,_orbital_inclination:float=0) -> bool:

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
					argperiapsis=_argperiapsis, orbital_inclination=_orbital_inclination, name=_name)
				setattr(self,_name,temp)
				return True
			elif check2:
				temp = planet(mass=_mass,radius=_radius,initial_velocity=_initial_velocity,initial_position=_initial_position)
				setattr(self,_name,temp)
				return True