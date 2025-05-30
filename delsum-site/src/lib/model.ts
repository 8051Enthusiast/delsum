export enum FieldType {
    Decimal,
    Hexadecimal,
    String,
    Endian,
    Wordsize,
    Boolean,
    Signedness,
}

export interface ModelField {
    name: string,
    type: FieldType,
}

export interface ModelDefinition {
    name: string,
    rows: ModelField[],
}

const wordspec: ModelField[] = [
    {
        name: "in_endian",
        type: FieldType.Endian
    },
    {
        name: "out_endian",
        type: FieldType.Endian
    },
    {
        name: "wordsize",
        type: FieldType.Wordsize
    }
]

const width: ModelField = {
    name: "width",
    type: FieldType.Decimal
}

const modsum: ModelDefinition = {
    name: "modsum",
    rows: [
        width,
        {
            name: "modulus",
            type: FieldType.Hexadecimal
        },
        {
            name: "init",
            type: FieldType.Hexadecimal
        },
        {
            name: "negated",
            type: FieldType.Boolean
        },
        {
            name: "signedness",
            type: FieldType.Signedness
        },
        ...wordspec
    ]
};

const fletcher: ModelDefinition = {
    name: "fletcher",
    rows: [
        width,
        {
            name: "modulus",
            type: FieldType.Hexadecimal
        },
        {
            name: "init",
            type: FieldType.Hexadecimal
        },
        {
            name: "addout",
            type: FieldType.Hexadecimal
        },
        {
            name: "swap",
            type: FieldType.Boolean
        },
        {
            name: "signedness",
            type: FieldType.Signedness
        },
        ...wordspec
    ]
};

const crc: ModelDefinition = {
    name: "crc",
    rows: [
        width,
        {
            name: "poly",
            type: FieldType.Hexadecimal
        },
        {
            name: "init",
            type: FieldType.Hexadecimal
        },
        {
            name: "xorout",
            type: FieldType.Hexadecimal
        },
        {
            name: "refin",
            type: FieldType.Boolean
        },
        {
            name: "refout",
            type: FieldType.Boolean
        },
        ...wordspec
    ]
}

const polyhash: ModelDefinition = {
    name: "polyhash",
    rows: [
        width,
        {
            name: "factor",
            type: FieldType.Hexadecimal
        },
        {
            name: "init",
            type: FieldType.Hexadecimal
        },
        {
            name: "addout",
            type: FieldType.Hexadecimal
        },
        {
            name: "signedness",
            type: FieldType.Signedness
        },
        ...wordspec
    ]
}

export const models: Record<string, ModelDefinition> = {
    "crc": crc,
    "fletcher": fletcher,
    "modsum": modsum,
    "polyhash": polyhash
}

export function modelValuesToString(model: ModelDefinition, values: Record<string, string>) {
    return model.rows.reduce((acc, row) => {
        let value = values[row.name];
        if (value != undefined && value != null && value != "") {
            if (row.type === FieldType.String) {
                value = `"${value}"`;
            }
            return `${acc} ${row.name}=${value}`;
        } else {
            return acc;
        }
    }, model.name);
}

export function stringToModelValue(modelString: string): [ModelDefinition, Record<string, string>] {
    const parts = modelString.trim().split(' ');
    const algorithm = parts[0];
    if (!models[algorithm]) {
        throw new Error(`Invalid model algorithm: ${modelString}`);
    }
    const definition = models[parts[0]];
    const values: Record<string, string> = {};
    for (let i = 1; i < parts.length; i++) {
        const part = parts[i];
        if (!part) {
            continue;
        }
        const [key, value] = part.split('=');
        if (!key || !value) {
            throw new Error(`Invalid model string: ${modelString}`);
        }
        values[key] = value;
    }
    return [definition, values];
}