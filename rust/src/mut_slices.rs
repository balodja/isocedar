pub struct MutSlices<'sl, 'off, T> {
    slice: &'sl mut [T],
    offsets: &'off [usize],
    current: usize,
}

impl<'sl, 'off, T> MutSlices<'sl, 'off, T> {
    pub fn new(slice: &'sl mut [T], offsets: &'off [usize]) -> Self {
        MutSlices {
            slice,
            offsets,
            current: 0,
        }
    }
}

impl<'sl, 'off, T> Iterator for MutSlices<'sl, 'off, T> {
    type Item = &'sl mut [T];

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.offsets.len() - 1 {
            return None;
        }

        let start = self.offsets[self.current];
        let end = self.offsets[self.current + 1];
        self.current += 1;

        // Here we temporarily replace slice with an empty one
        let (left, right) = std::mem::take(&mut self.slice).split_at_mut(end - start);
        self.slice = right;
        Some(left)
    }
}
