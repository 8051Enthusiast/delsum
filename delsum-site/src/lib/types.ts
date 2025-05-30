export type FileRowContent<T> = { id: number, file: T; checksum: string };
export function containsString(row: FileRowContent<string | File>): row is FileRowContent<string> {
    return typeof row.file === "string";
}
export function containsFile(row: FileRowContent<string | File>): row is FileRowContent<File> {
    return typeof row.file === "object";
}
