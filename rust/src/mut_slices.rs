use rayon::iter::plumbing::{bridge, Consumer, Producer, ProducerCallback, UnindexedConsumer};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};

pub struct MutSlices<'sl, 'off, T> {
    slice: &'sl mut [T],
    offsets: &'off [usize],
}

impl<'sl, 'off, T> MutSlices<'sl, 'off, T> {
    pub fn new(slice: &'sl mut [T], offsets: &'off [usize]) -> Self {
        MutSlices { slice, offsets }
    }
}

impl<'sl, 'off, T> Iterator for MutSlices<'sl, 'off, T> {
    type Item = &'sl mut [T];

    fn next(&mut self) -> Option<Self::Item> {
        if self.offsets.len() <= 1 {
            return None;
        }

        // Split slice, save right. Here we temporarily replace slice with an empty one
        let (left, right) =
            std::mem::take(&mut self.slice).split_at_mut(self.offsets[1] - self.offsets[0]);
        self.slice = right;

        // Move offsets to the right
        self.offsets = &self.offsets[1..];

        Some(left)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.offsets.len() - 1;
        (len, Some(len))
    }
}

impl<'sl, 'off, T> DoubleEndedIterator for MutSlices<'sl, 'off, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let offsets_len = self.offsets.len();
        if offsets_len <= 1 {
            return None;
        }

        // Split slice, save left. Here we temporarily replace slice with an empty one
        let (left, right) = std::mem::take(&mut self.slice)
            .split_at_mut(self.offsets[offsets_len - 2] - self.offsets[0]);
        self.slice = left;

        // Move offsets to the left
        self.offsets = &self.offsets[..offsets_len - 1];

        Some(right)
    }
}

impl<'sl, 'off, T> ExactSizeIterator for MutSlices<'sl, 'off, T> {
    fn len(&self) -> usize {
        self.offsets.len() - 1
    }
}

// Following rayon's ChunksMut implementation
impl<'sl, 'off, T> ParallelIterator for MutSlices<'sl, 'off, T>
where
    T: Send,
{
    type Item = &'sl mut [T];

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        ExactSizeIterator::len(self).into()
    }
}

impl<'sl, 'off, T> IndexedParallelIterator for MutSlices<'sl, 'off, T>
where
    T: Send,
{
    fn len(&self) -> usize {
        ExactSizeIterator::len(self)
    }

    fn drive<C>(self, consumer: C) -> C::Result
    where
        C: Consumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn with_producer<CB>(self, callback: CB) -> CB::Output
    where
        CB: ProducerCallback<Self::Item>,
    {
        callback.callback(MutSlicesProducer {
            slice: self.slice,
            offsets: self.offsets,
        })
    }
}

struct MutSlicesProducer<'sl, 'off, T> {
    slice: &'sl mut [T],
    offsets: &'off [usize],
}

impl<'sl, 'off, T> Producer for MutSlicesProducer<'sl, 'off, T>
where
    T: Send,
{
    type Item = &'sl mut [T];
    type IntoIter = MutSlices<'sl, 'off, T>;

    fn into_iter(self) -> Self::IntoIter {
        MutSlices::new(self.slice, self.offsets)
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let (left_slice, right_slice) = self
            .slice
            .split_at_mut(self.offsets[index] - self.offsets[0]);
        let (left_offsets, right_offsets) = (&self.offsets[..=index], &self.offsets[index..]);
        (
            MutSlicesProducer {
                slice: left_slice,
                offsets: left_offsets,
            },
            MutSlicesProducer {
                slice: right_slice,
                offsets: right_offsets,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use itertools::Itertools;

    #[test]
    fn test_mut_slices() {
        let mut data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let offsets = vec![0, 3, 6, 10];

        let mut slices = MutSlices::new(&mut data, &offsets);

        assert_eq!(slices.next().unwrap(), &[1, 2, 3]);
        assert_eq!(slices.next().unwrap(), &[4, 5, 6]);
        assert_eq!(slices.next().unwrap(), &[7, 8, 9, 10]);
        assert!(slices.next().is_none());

        let mut slices = MutSlices::new(&mut data, &offsets);

        assert_eq!(slices.next_back().unwrap(), &[7, 8, 9, 10]);
        assert_eq!(slices.next_back().unwrap(), &[4, 5, 6]);
        assert_eq!(slices.next_back().unwrap(), &[1, 2, 3]);
        assert!(slices.next_back().is_none());

        let mut slices = MutSlices::new(&mut data, &offsets);
        assert_eq!(slices.next().unwrap(), &[1, 2, 3]);
        assert_eq!(slices.next_back().unwrap(), &[7, 8, 9, 10]);
        assert_eq!(slices.next().unwrap(), &[4, 5, 6]);
        assert_eq!(slices.next_back(), None);
        assert_eq!(slices.next(), None);
    }

    #[test]
    fn test_mut_slices_len() {
        let mut data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let offsets = vec![0, 3, 6, 10];

        let slices = MutSlices::new(&mut data, &offsets);

        assert_eq!(ExactSizeIterator::len(&slices), 3);
    }

    #[test]
    fn test_mut_slices_parallel() {
        let mut data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let offsets = vec![0, 3, 6, 10];

        let slices = MutSlices::new(&mut data, &offsets);

        let sum: usize = ParallelIterator::map(slices, |slice| slice.iter().sum::<usize>()).sum();

        assert_eq!(sum, data.iter().sum());
    }

    #[test]
    fn test_mut_slices_producer() {
        let mut data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let offsets = vec![0, 3, 6, 10];

        let producer = MutSlicesProducer {
            slice: &mut data,
            offsets: &offsets,
        };
        assert_eq!(
            producer.into_iter().collect_vec(),
            [vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9, 10]]
        );

        let producer = MutSlicesProducer {
            slice: &mut data,
            offsets: &offsets,
        };
        let (left, right) = producer.split_at(1);
        assert_eq!(left.into_iter().collect_vec(), [&[1, 2, 3]]);
        assert_eq!(
            right.into_iter().collect_vec(),
            [&vec![4, 5, 6], &vec![7, 8, 9, 10]]
        );
    }
}
