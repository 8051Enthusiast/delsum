export function replaceValue<Row extends { id: number }>(newInput: Row, values: Row[]) {
    return values.map((item) => {
        if (item.id == newInput.id) {
            return newInput;
        } else {
            return item;
        }
    });
}

export function removeValue<Row extends { id: number }>(id: number, values: Row[]) {
    return values.filter((item) => item.id != id);
}