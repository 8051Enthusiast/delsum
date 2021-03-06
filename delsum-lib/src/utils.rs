/// Turns Result<Iterator, Error> into Iterator<Result<Iterator::Item, Error>> so that
/// on Err, only the single error is iterated, and else the items of the iterator
pub fn unresult_iter<I, E>(x: Result<I, E>) -> impl Iterator<Item = Result<I::Item, E>>
where
    I: std::iter::Iterator,
    E: std::error::Error,
{
    let (i, e) = match x {
        Ok(i) => (Some(i.map(Ok)), None),
        Err(e) => (None, Some(std::iter::once(Err(e)))),
    };
    e.into_iter().flatten().chain(i.into_iter().flatten())
}

/// Small function for small cartesian products of small data types
pub fn cart_prod<T: Clone, U: Clone>(a: &[T], b: &[U]) -> Vec<(T, U)> {
    let mut v = Vec::new();
    for x in a {
        for y in b {
            v.push((x.clone(), y.clone()))
        }
    }
    v
}