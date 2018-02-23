#-------------------------------------------------------------------------
# . NexusHandler

def parseInt(number):
    try:
        return int(number)
    except ValueError as e:
        raise Exception("Invalid scan numbers: %s" % str(e))

    return 0


def procNumbers(numberList):
    if len(numberList) == 0 or numberList == '0':
        return list()  # this is what is expected elsewhere
    numberList = [str(scan) for scan in [numberList]]
    numberList = [num for num in str(','.join(numberList)).split(',')]

    result = []
    if isinstance(numberList, str):
        if "-" in numberList:
            item = [parseInt(i) for i in numberList.split("-")]
            if item[0] is not None:
                result.extend(range(item[0], item[1] + 1))

    else:
        for item in numberList:
            # if there is a dash then it is a range
            if "-" in item:
                item = sorted([parseInt(i) for i in item.split("-")])
                if item[0] is not None:
                    result.extend(range(item[0], item[1] + 1))
            else:
                item = parseInt(item)
                if item:
                    result.append(item)

    result.sort()
    return result


class NexusHandler(object):
    def __init__(self, instrument, cfg_filename):
        self.instrument = instrument
        self._scanDict = {}

        config_path = os.path.join('/SNS', instrument, 'shared', cfg_filename)
        config = configparser.ConfigParser()
        config.read(config_path)
        self._props = {name: path for name, path in config.items('meta')}
        self._props.update(
            {name: path for name, path in config.items('nexus')})

    def listProps(self):
        return self._props.keys()

    def getNxData(self, scans, props):
        scansInfo = dict()
        for scan in scans:
            # convert to format for mantid's file finder
            filename = '%s_%s' % (self.instrument, scan)
            # let mantid find the file
            filename = mantid.api.FileFinder.findRuns(filename)[0]
            # get properties specified in the config file
            with File(filename, 'r') as nf:
                prop_dict = {prop: self._props[prop] for prop in props}
                for key, path in prop_dict.items(
                ):  # inefficient in py2, but works with py3
                    try:
                        scansInfo.update({key: nf[path][0]})
                    except KeyError:
                        pass
        return scansInfo

#-------------------------------------------------------------------------
# Absolute Scale stuff


def combine_dictionaries(*dictionaries):
    result = dict()
    for dictionary in dictionaries:
        for key, values in dictionary.items():
            if key in result:
                result[key].update(values)
            else:
                result[key] = values
    return result


class Shape(object):
    def __init__(self):
        self.shape = None

    def getShape(self):
        return self.shape


class Cylinder(Shape):
    def __init__(self):
        self.shape = 'Cylinder'

    def volume(self, Radius=None, Height=None):
        return np.pi * Height * Radius * Radius


class Sphere(Shape):
    def __init__(self):
        self.shape = 'Sphere'

    def volume(self, Radius=None):
        return (4. / 3.) * np.pi * Radius * Radius * Radius


class GeometryFactory(object):

    @staticmethod
    def factory(Geometry):
        factory = {"Cylinder": Cylinder(),
                   "Sphere": Sphere()}
        return factory[Geometry["Shape"]]


def getAbsScaleInfoFromNexus(
        scans,
        ChemicalFormula=None,
        Geometry=None,
        PackingFraction=None,
        BeamWidth=1.8,
        SampleMassDensity=None):
    # get necessary properties from Nexus file
    props = [
        "formula",
        "mass",
        "mass_density",
        "sample_diameter",
        "sample_height",
        "items_id"]
    info = nf.getNxData(scans, props)
    info['sample_diameter'] = 0.1 * float(info['sample_diameter'])  # mm -> cm

    for key in info:
        print(key, info[key])

    if ChemicalFormula:
        info["formula"] = ChemicalFormula
    if SampleMassDensity:
        info["mass_density"] = float(SampleMassDensity)

    # setup the geometry of the sample
    if Geometry is None:
        Geometry = dict()
    if "Shape" not in Geometry:
        Geometry["Shape"] = 'Cylinder'
    if "Radius" not in Geometry:
        Geometry['Radius'] = info['sample_diameter'] / 2.
    if "Height" not in Geometry:
        Geometry['Height'] = info['sample_height']

    if Geometry["Shape"] == 'Sphere':
        Geometry.pop('Height', None)

    # get sample volume in container
    space = GeometryFactory.factory(Geometry)
    Geometry.pop("Shape", None)
    volume_in_container = space.volume(**Geometry)

    try:
        print(
            "NeXus Packing Fraction:",
            info["mass"] /
            volume_in_container /
            info["mass_density"])
    except BaseException:
        print("NeXus Packing Fraction - not calculatable")
    # get packing fraction
    if PackingFraction is None:
        sample_density = info["mass"] / volume_in_container
        PackingFraction = sample_density / info["mass_density"]

    info['packing_fraction'] = PackingFraction

    print("PackingFraction:", PackingFraction)

    # get sample volume in the beam and correct mass density of what is in the
    # beam
    if space.getShape() == 'Cylinder':
        Geometry["Height"] = BeamWidth
    volume_in_beam = space.volume(**Geometry)
    print(info["mass_density"], PackingFraction)
    mass_density_in_beam = PackingFraction * info["mass_density"]

    # get molecular mass
    # Mantid SetSample doesn't set the actual height or radius. Have to use
    # the setHeight, setRadius, ....
    ws = CreateSampleWorkspace()
    # SetSample(ws, Geometry={"Shape" : "Cylinder", "Height" : geo_dict["height"], "Radius" : geo_dict["radius"], "Center" : [0.,0.,0.]},
    # Material={"ChemicalFormula" : info["formula"], "SampleMassDensity" :
    # PackingFraction * info["mass_density"]})

    print(info["formula"], mass_density_in_beam, volume_in_beam)
    if not info["formula"] or info["formula"] == 'N/A':
        return [None, None, None]

    SetSampleMaterial(
        ws,
        ChemicalFormula=info["formula"],
        SampleMassDensity=mass_density_in_beam)
    material = ws.sample().getMaterial()

    # set constant
    avogadro = 6.022 * 10**23.

    # get total atoms and individual atom info
    natoms = sum([x for x in material.chemicalFormula()[1]])
    concentrations = {
        atom.symbol: {
            'concentration': conc,
            'mass': atom.mass} for atom,
        conc in zip(
            material.chemicalFormula()[0],
            material.chemicalFormula()[1])}
    neutron_info = {atom.symbol: atom.neutron()
                    for atom in material.chemicalFormula()[0]}
    atoms = combine_dictionaries(concentrations, neutron_info)

    sigfree = [atom['tot_scatt_xs'] * atom['concentration'] * \
        (atom['mass'] / (atom['mass'] + 1.0))**2. for atom in atoms.values()]
    print(sum(sigfree))

    # get number of atoms using packing fraction, density, and volume
    print(
        "Total scattering Xsection",
        material.totalScatterXSection() *
        natoms)
    print("Coh Xsection:", material.cohScatterXSection() * natoms)
    print("Incoh Xsection:", material.incohScatterXSection() * natoms)
    print("Abs. Xsection:", material.absorbXSection() * natoms)

    print(''.join([x.strip() for x in info["formula"]]), "#sample title")
    print(info["formula"], "#sample formula")
    print(info["mass_density"], "#density")
    print(Geometry["Radius"], "#radius")
    print(PackingFraction, "#PackingFraction")
    print(space.getShape(), "#sample shape")
    print("nogo", "#do absorption correction now")
    print(info["mass_density"] / material.relativeMolecularMass() *
          avogadro / 10**24., "Sample density in form unit / A^3")

    print("\n\n#########################################################")
    print("##############Check levels###########################################")
    print("b bar:", material.cohScatterLengthReal())
    print("sigma:", material.totalScatterXSection())
    print("b: ", np.sqrt(material.totalScatterXSection() / (4. * np.pi)))
    print(
        material.cohScatterLengthReal() *
        material.cohScatterLengthReal() *
        natoms *
        natoms,
        "# (sum b)^2")
    print(material.cohScatterLengthSqrd() * natoms, "# (sum c*bbar^2)")
    self_scat = material.totalScatterLengthSqrd() * natoms / \
        100.  # 100 fm^2 == 1 barn
    print("self scattering:", self_scat)
    print("#########################################################\n")

    natoms_in_beam = mass_density_in_beam / material.relativeMolecularMass() * \
        avogadro / 10**24. * volume_in_beam
    #print("Sample density (corrected) in form unit / A^3: ", mass_density_in_beam/ ws.sample().getMaterial().relativeMolecularMass() * avogadro / 10**24.)
    return natoms_in_beam, self_scat, info

