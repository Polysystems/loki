// Quick test to verify lockfree compilation
use std::marker::PhantomData;

// Test 1: PhantomData in struct initialization
struct TestRef<K, V> {
    value: V,
    _phantom: PhantomData<K>,
}

impl<K, V> TestRef<K, V> {
    fn new(value: V) -> Self {
        Self {
            value,
            _phantom: PhantomData,
        }
    }
}

// Test 2: Wide crate API
#[cfg(feature = "wide")]
fn test_wide() {
    use wide::f32x8;
    
    let a: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let vec = f32x8::from(a);
    let arr: [f32; 8] = vec.to_array();
    let sum = vec.reduce_sum();
    
    println!("Array: {:?}, Sum: {}", arr, sum);
}

// Test 3: ArcSwapOption load
#[cfg(feature = "arc-swap")]
fn test_arcswap() {
    use arc_swap::ArcSwapOption;
    use std::sync::Arc;
    
    let swap = ArcSwapOption::from(Some(Arc::new(42)));
    let guard = swap.load();
    if guard.is_some() {
        println!("Value: {:?}", guard.as_ref());
    }
}

fn main() {
    // Test PhantomData initialization
    let test_ref = TestRef::new(42);
    println!("TestRef created with value: {}", test_ref.value);
    
    #[cfg(feature = "wide")]
    test_wide();
    
    #[cfg(feature = "arc-swap")]
    test_arcswap();
    
    println!("All tests passed!");
}