import itertools
import json
import base64
import pathlib
import uuid
from collections import defaultdict

import cfgrib
import eccodes
import numpy as np

from .magician import Magician

import logging

logger = logging.getLogger("gribscan")

def _split_file(f, skip=0):
    """
    splits a gribfile into individual messages
    """
    if hasattr(f, "size"):
        size = f.size
    else:
        f.seek(0, 2)
        size = f.tell()
        f.seek(0)
    part = 0

    while f.tell() < size:
        logger.debug(f"extract part {part + 1}")
        start = f.tell()
        f.seek(12, 1)
        part_size = int.from_bytes(f.read(4), "big")
        f.seek(start)
        data = f.read(part_size)
        assert data[:4] == b"GRIB"
        assert data[-4:] == b"7777"
        yield start, part_size, data
        part += 1
        if skip and part > skip:
            break

EXTRA_PARAMETERS = [
    "forecastTime",
    "indicatorOfUnitOfTimeRange",
    "lengthOfTimeRange",
    "indicatorOfUnitForTimeRange",
    "productDefinitionTemplateNumber",
]

production_template_numbers = {
    0: {"forcastTime": True, "timeRange": False},
    1: {"forcastTime": True, "timeRange": False},
    2: {"forcastTime": True, "timeRange": False},
    3: {"forcastTime": True, "timeRange": False},
    4: {"forcastTime": True, "timeRange": False},
    5: {"forcastTime": True, "timeRange": False},
    6: {"forcastTime": True, "timeRange": False},
    7: {"forcastTime": True, "timeRange": False},
    15: {"forcastTime": True, "timeRange": False},
    32: {"forcastTime": True, "timeRange": False},
    33: {"forcastTime": True, "timeRange": False},
    40: {"forcastTime": True, "timeRange": False},
    41: {"forcastTime": True, "timeRange": False},
    44: {"forcastTime": True, "timeRange": False},
    45: {"forcastTime": True, "timeRange": False},
    48: {"forcastTime": True, "timeRange": False},
    51: {"forcastTime": True, "timeRange": False},
    53: {"forcastTime": True, "timeRange": False},
    54: {"forcastTime": True, "timeRange": False},
    55: {"forcastTime": True, "timeRange": False},
    56: {"forcastTime": True, "timeRange": False},
    57: {"forcastTime": True, "timeRange": False},
    58: {"forcastTime": True, "timeRange": False},
    60: {"forcastTime": True, "timeRange": False},
    1000: {"forcastTime": True, "timeRange": False},
    1002: {"forcastTime": True, "timeRange": False},
    1100: {"forcastTime": True, "timeRange": False},
    40033: {"forcastTime": True, "timeRange": False},
    40455: {"forcastTime": True, "timeRange": False},
    40456: {"forcastTime": True, "timeRange": False},

    20: {"forcastTime": False, "timeRange": False},
    30: {"forcastTime": False, "timeRange": False},
    31: {"forcastTime": False, "timeRange": False},
    254: {"forcastTime": False, "timeRange": False},
    311: {"forcastTime": False, "timeRange": False},
    2000: {"forcastTime": False, "timeRange": False},

    8: {"forcastTime": True, "timeRange": True},
    9: {"forcastTime": True, "timeRange": True},
    10: {"forcastTime": True, "timeRange": True},
    11: {"forcastTime": True, "timeRange": True},
    12: {"forcastTime": True, "timeRange": True},
    13: {"forcastTime": True, "timeRange": True},
    14: {"forcastTime": True, "timeRange": True},
    34: {"forcastTime": True, "timeRange": True},
    42: {"forcastTime": True, "timeRange": True},
    43: {"forcastTime": True, "timeRange": True},
    46: {"forcastTime": True, "timeRange": True},
    47: {"forcastTime": True, "timeRange": True},
    61: {"forcastTime": True, "timeRange": True},
    67: {"forcastTime": True, "timeRange": True},
    68: {"forcastTime": True, "timeRange": True},
    91: {"forcastTime": True, "timeRange": True},
    1001: {"forcastTime": True, "timeRange": True},
    1101: {"forcastTime": True, "timeRange": True},
    10034: {"forcastTime": True, "timeRange": True},
}

# according to http://www.cosmo-model.org/content/consortium/generalMeetings/general2014/wg6-pompa/grib2/grib/pdtemplate_4.41.htm
time_range_units = {
    0: 60,  # np.timedelta64(1, "m"),
    1: 60*60,  # np.timedelta64(1, "h"),
    2: 24*60*60,  # np.timedelta64(1, "D"),
    #3   Month
    #4   Year
    #5   Decade (10 years)
    #6   Normal (30 years)
    #7   Century (100 years)
    #8-9 Reserved
    10: 3*60*60,  # np.timedelta64(3, "h"),
    11: 6*60*60,  # np.timedelta64(6, "h"),
    12: 12*60*60,  # np.timedelta64(12, "h"),
    13: 1,  # np.timedelta64(1, "s"),
    #14-191  Reserved
    #192-254 Reserved for local use
    #255 Missing
}

def get_time_offset(gribmessage):
    offset = 0  # np.timedelta64(0, "s")
    try:
        options = production_template_numbers[int(gribmessage["productDefinitionTemplateNumber"])]
    except KeyError:
        return offset
    if options["forcastTime"]:
        unit = time_range_units[int(gribmessage.get("indicatorOfUnitOfTimeRange", 255))]
        offset += gribmessage.get("forecastTime", 0) * unit
    # TODO: handling of time ranges, see cdo: libcdi/src/gribapi_utilities.c: gribMakeTimeString
    return offset


def scan_gribfile(filelike, **kwargs):
    for offset, size, data in _split_file(filelike):
        mid = eccodes.codes_new_from_message(data)
        m = cfgrib.cfmessage.CfMessage(mid)
        t = eccodes.codes_get_native_type(m.codes_id, "values")
        s = eccodes.codes_get_size(m.codes_id, "values")

        hgrid_uuid = uuid.UUID(eccodes.codes_get_string(mid, "uuidOfHGrid"))
        yield {"globals": {
                   **{k: m[k] for k in cfgrib.dataset.GLOBAL_ATTRIBUTES_KEYS},
                   "uuidOfHGrid": str(hgrid_uuid),
               },
               "attrs": {k: m.get(k, None) for k in cfgrib.dataset.DATA_ATTRIBUTES_KEYS + cfgrib.dataset.EXTRA_DATA_ATTRIBUTES_KEYS},
               "parameter_code": {
                   k: m.get(k, None)
                   for k in ["discipline", "parameterCategory", "parameterNumber"]
                },
               "posix_time": m["time"] + get_time_offset(m),
               "domain": m["globalDomain"],
               "time": f"{m['hour']:02d}{m['minute']:02d}",
               "date": f"{m['year']:04d}{m['month']:02d}{m['day']:02d}",
               "levtype": m.get("typeOfLevel", None),
               "level": m.get("level", None),
               "param": m.get("shortName", None),
               "type": m.get("dataType", None),
               "referenceTime": m["time"],
               "step": m["step"],
               "_offset": offset,
               "_length": size,
               "array": {
                   "dtype": np.dtype(t).str,
                   "shape": [s],
                },
               "extra": {k: m.get(k, None) for k in EXTRA_PARAMETERS},
               **kwargs
               }


def write_index(gribfile, idxfile=None):
    p = pathlib.Path(gribfile)
    if idxfile is None:
        idxfile = p.parent / (p.stem + ".index")

    gen = scan_gribfile(open(p, "rb"), filename=p.name)

    with open(idxfile, 'w') as output_file:
        for record in gen:
            json.dump(record, output_file)
            output_file.write("\n")

def parse_index(indexfile, m2key, duplicate="replace"):
    index = {}
    with open(indexfile, "r") as f:
        for line in f:
            meta = json.loads(line)
            tinfo = m2key(meta)
            if tinfo in index:
                if duplicate == "replace":
                    index[tinfo] = meta
                elif duplicate == "keep":
                    continue
                elif duplicate == "error":
                    raise Exception(f"Duplicate message step: {tinfo}")
            else:
                index[tinfo] = meta
    return list(index.values())

def inspect_grib_indices(messages, magician):
    coords_by_key = defaultdict(lambda: tuple(set() for _ in magician.dimkeys))
    size_by_key = defaultdict(set)
    attrs_by_key = {}
    dtype_by_key = {}
    global_attrs = {}

    for msg in messages:
        varkey, coords = magician.m2key(msg)
        for existing, new in zip(coords_by_key[varkey], coords):
            existing.add(new)
        size_by_key[varkey].add(msg["array"]["shape"][0])
        attrs_by_key[varkey] = {k: v for k, v in msg["attrs"].items()
                                if v is not None and v not in {"undef", "unknown"}}
        dtype_by_key[varkey] = msg["array"]["dtype"]
        global_attrs = msg["globals"]

    for k, v in size_by_key.items():
        assert len(v) == 1, f"inconsistent shape of {k}"

    size_by_key = {k: list(v)[0] for k, v in size_by_key.items()}

    varinfo = {}
    for varkey, coords in coords_by_key.items():
        dims, dim_id, shape = map(tuple, zip(*((dim, i, len(coords))
                                               for i, (dim, coords) in enumerate(zip(magician.dimkeys, coords))
                                               if len(coords) != 1)))
        info = {
            "dims": dims,
            "shape": shape,
            "dim_id": dim_id,
            "coords": tuple(coords_by_key[varkey][i] for i in dim_id),
            "data_size": size_by_key[varkey],
            "data_dim": "cell",
            "dtype": dtype_by_key[varkey],
            "attrs": attrs_by_key[varkey],
        }
        varinfo[varkey] = {
            **info,
            **magician.variable_hook(varkey, info),
        }

    coords = defaultdict(set)
    for _, info in varinfo.items():
        for dim, cs in zip(info["dims"], info["coords"]):
            coords[dim] |= cs

    coords = {k: list(sorted(c)) for k, c in coords.items()}

    return global_attrs, coords, varinfo

def build_refs(messages, global_attrs, coords, varinfo, magician):
    coords_inv = {k: {v: i for i, v in enumerate(vs)} for k, vs in coords.items()}

    refs = {}
    for msg in messages:
        key, coord = magician.m2key(msg)
        info = varinfo[key]
        cs = [coord[d] for d in info["dim_id"]]
        chunk_id = ".".join(map(str, [coords_inv[d][c] for d, c in zip(info["dims"], cs)])) + ".0"
        refs[info["name"] + "/" + chunk_id] = [msg["filename"], msg["_offset"], msg["_length"]]

    for varkey, info in varinfo.items():
        refs[info["name"] + "/.zattrs"] = json.dumps({**info["attrs"],
                                                      "_ARRAY_DIMENSIONS": list(info["dims"]) + [info["data_dim"]]})
        shape = [len(coords[dim]) for dim in info["dims"]] + [info["data_size"]]
        chunks = [1 for _ in info["shape"]] + [info["data_size"]]
        refs[info["name"] + "/.zarray"] = json.dumps({
            "shape": shape,
            "chunks": chunks,
            "compressor": {"id": "gribscan.rawgrib"},
            "dtype": info["dtype"],
            "fill_value": info["attrs"].get("missingValue", 9999),
            "filters": [],
            "order": "C",
            "zarr_format": 2,
        })

    for name, cs in coords.items():
        cs = np.asarray(cs)
        attrs, cs = magician.coords_hook(name, cs)
        refs[f"{name}/.zattrs"] = json.dumps({**attrs, "_ARRAY_DIMENSIONS": [name]})
        refs[f"{name}/.zarray"] = json.dumps({
            "chunks": [cs.size],
            "compressor": None,
            "dtype": cs.dtype.str,
            "fill_value": 0,
            "filters": [],
            "order": "C",
            "shape": [cs.size],
            "zarr_format": 2,
        })
        refs[f"{name}/0"] = "base64:" + base64.b64encode(bytes(cs)).decode("ascii")

    refs[".zgroup"] = json.dumps({"zarr_format": 2})
    refs[".zattrs"] = json.dumps(magician.globals_hook(global_attrs))

    return refs

def compress_ref_keys(refs):
    table = {f: f"f{i}" for i, f in enumerate(sorted(set(target[0] for target in refs.values() if isinstance(target, list))))}
    refs = {k: ["{{u}}{{" + table[target[0]] + "}}"] + target[1:] if isinstance(target, list) else target
            for k, target in refs.items()}
    return refs, {v: k for k, v in table.items()}

def grib_magic(filenames, magician=None, global_prefix=""):
    if magician is None:
        magician = Magician()

    messages = [msg
                for filename in filenames
                for msg in parse_index(filename, magician.m2key)]

    global_attrs, coords, varinfo = inspect_grib_indices(messages, magician)
    refs = build_refs(messages, global_attrs, coords, varinfo, magician)
    short_refs, table = compress_ref_keys(refs)
    return {
        "version": 1,
        "templates": {
           "u": global_prefix,  # defaults to plain filenames (can be updated externally)
            **table,
        },
        "refs": short_refs
    }
