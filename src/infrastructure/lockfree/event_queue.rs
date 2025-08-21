//! Lock-free event queue implementation for high-performance event processing

use crossbeam_queue::ArrayQueue;
use crossbeam_channel::{unbounded, Sender, Receiver};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::{Duration, Instant};
use std::collections::BinaryHeap;
use std::cmp::Ordering as CmpOrdering;
use dashmap::DashMap;
use bytes::Bytes;

/// Event priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EventPriority {
    Critical = 0,
    High = 1,
    Normal = 2,
    Low = 3,
}

/// Generic event structure with zero-copy payload
#[derive(Clone, Debug)]
pub struct Event {
    pub id: u64,
    pub timestamp: Instant,
    pub priority: EventPriority,
    pub topic: String,
    pub payload: Bytes, // Zero-copy payload
    pub metadata: Arc<EventMetadata>,
}

impl Event {
    pub fn new(topic: String, payload: Bytes, priority: EventPriority) -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        
        Self {
            id: COUNTER.fetch_add(1, Ordering::Relaxed),
            timestamp: Instant::now(),
            priority,
            topic,
            payload,
            metadata: Arc::new(EventMetadata::default()),
        }
    }
    
    pub fn with_metadata(mut self, metadata: EventMetadata) -> Self {
        self.metadata = Arc::new(metadata);
        self
    }
}

/// Event metadata
#[derive(Debug, Clone, Default)]
pub struct EventMetadata {
    pub source: Option<String>,
    pub correlation_id: Option<String>,
    pub trace_id: Option<String>,
    pub retry_count: usize,
}

/// Lock-free event queue with priority support
pub struct LockFreeEventQueue {
    // Segmented queues by priority
    critical_queue: Arc<ArrayQueue<Event>>,
    high_queue: Arc<ArrayQueue<Event>>,
    normal_queue: Arc<ArrayQueue<Event>>,
    low_queue: Arc<ArrayQueue<Event>>,
    
    // Statistics
    events_processed: AtomicU64,
    events_dropped: AtomicU64,
    
    // Queue state
    closed: AtomicBool,
}

impl LockFreeEventQueue {
    pub fn new(capacity_per_priority: usize) -> Self {
        Self {
            critical_queue: Arc::new(ArrayQueue::new(capacity_per_priority)),
            high_queue: Arc::new(ArrayQueue::new(capacity_per_priority)),
            normal_queue: Arc::new(ArrayQueue::new(capacity_per_priority)),
            low_queue: Arc::new(ArrayQueue::new(capacity_per_priority)),
            events_processed: AtomicU64::new(0),
            events_dropped: AtomicU64::new(0),
            closed: AtomicBool::new(false),
        }
    }
    
    /// Push event to appropriate priority queue
    pub fn push(&self, event: Event) -> Result<(), Event> {
        if self.closed.load(Ordering::Acquire) {
            return Err(event);
        }
        
        super::GLOBAL_STATS.record_operation();
        
        let result = match event.priority {
            EventPriority::Critical => self.critical_queue.push(event),
            EventPriority::High => self.high_queue.push(event),
            EventPriority::Normal => self.normal_queue.push(event),
            EventPriority::Low => self.low_queue.push(event),
        };
        
        if result.is_err() {
            self.events_dropped.fetch_add(1, Ordering::Relaxed);
            super::GLOBAL_STATS.record_contention();
        }
        
        result
    }
    
    /// Pop event with priority order
    pub fn pop(&self) -> Option<Event> {
        super::GLOBAL_STATS.record_operation();
        
        // Try queues in priority order
        if let Some(event) = self.critical_queue.pop() {
            self.events_processed.fetch_add(1, Ordering::Relaxed);
            return Some(event);
        }
        
        if let Some(event) = self.high_queue.pop() {
            self.events_processed.fetch_add(1, Ordering::Relaxed);
            return Some(event);
        }
        
        if let Some(event) = self.normal_queue.pop() {
            self.events_processed.fetch_add(1, Ordering::Relaxed);
            return Some(event);
        }
        
        if let Some(event) = self.low_queue.pop() {
            self.events_processed.fetch_add(1, Ordering::Relaxed);
            return Some(event);
        }
        
        None
    }
    
    /// Try pop with timeout
    pub fn pop_timeout(&self, timeout: Duration) -> Option<Event> {
        let deadline = Instant::now() + timeout;
        
        loop {
            if let Some(event) = self.pop() {
                return Some(event);
            }
            
            if Instant::now() >= deadline {
                return None;
            }
            
            if self.closed.load(Ordering::Acquire) {
                return None;
            }
            
            std::hint::spin_loop();
        }
    }
    
    /// Check if all queues are empty
    pub fn is_empty(&self) -> bool {
        self.critical_queue.is_empty() &&
        self.high_queue.is_empty() &&
        self.normal_queue.is_empty() &&
        self.low_queue.is_empty()
    }
    
    /// Get total queue size
    pub fn len(&self) -> usize {
        self.critical_queue.len() +
        self.high_queue.len() +
        self.normal_queue.len() +
        self.low_queue.len()
    }
    
    /// Close the queue
    pub fn close(&self) {
        self.closed.store(true, Ordering::Release);
    }
    
    /// Get queue statistics
    pub fn stats(&self) -> EventQueueStats {
        EventQueueStats {
            events_processed: self.events_processed.load(Ordering::Relaxed),
            events_dropped: self.events_dropped.load(Ordering::Relaxed),
            critical_size: self.critical_queue.len(),
            high_size: self.high_queue.len(),
            normal_size: self.normal_queue.len(),
            low_size: self.low_queue.len(),
        }
    }
}

/// Event queue statistics
#[derive(Debug, Clone)]
pub struct EventQueueStats {
    pub events_processed: u64,
    pub events_dropped: u64,
    pub critical_size: usize,
    pub high_size: usize,
    pub normal_size: usize,
    pub low_size: usize,
}

/// Topic-based event router
pub struct EventRouter {
    // Topic to subscribers mapping
    subscribers: Arc<DashMap<String, Vec<Sender<Event>>>>,
    
    // Event queue
    queue: Arc<LockFreeEventQueue>,
    
    // Routing statistics
    routed: AtomicU64,
    failed: AtomicU64,
}

impl EventRouter {
    pub fn new(queue_capacity: usize) -> Self {
        Self {
            subscribers: Arc::new(DashMap::new()),
            queue: Arc::new(LockFreeEventQueue::new(queue_capacity)),
            routed: AtomicU64::new(0),
            failed: AtomicU64::new(0),
        }
    }
    
    /// Subscribe to a topic
    pub fn subscribe(&self, topic: String) -> Receiver<Event> {
        let (tx, rx) = unbounded();
        
        self.subscribers
            .entry(topic)
            .or_insert_with(Vec::new)
            .push(tx);
        
        rx
    }
    
    /// Publish event to topic
    pub fn publish(&self, event: Event) -> Result<(), Event> {
        // Add to queue first
        self.queue.push(event.clone())?;
        
        // Route to subscribers
        if let Some(subscribers) = self.subscribers.get(&event.topic) {
            for sender in subscribers.iter() {
                if sender.send(event.clone()).is_ok() {
                    self.routed.fetch_add(1, Ordering::Relaxed);
                } else {
                    self.failed.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        
        Ok(())
    }
    
    /// Process events from queue
    pub async fn process_events(&self) {
        while let Some(event) = self.queue.pop() {
            if let Some(subscribers) = self.subscribers.get(&event.topic) {
                for sender in subscribers.iter() {
                    let _ = sender.send(event.clone());
                }
            }
        }
    }
    
    /// Get routing statistics
    pub fn stats(&self) -> RouterStats {
        RouterStats {
            topics: self.subscribers.len(),
            total_subscribers: self.subscribers.iter()
                .map(|entry| entry.value().len())
                .sum(),
            events_routed: self.routed.load(Ordering::Relaxed),
            events_failed: self.failed.load(Ordering::Relaxed),
            queue_stats: self.queue.stats(),
        }
    }
}

/// Router statistics
#[derive(Debug, Clone)]
pub struct RouterStats {
    pub topics: usize,
    pub total_subscribers: usize,
    pub events_routed: u64,
    pub events_failed: u64,
    pub queue_stats: EventQueueStats,
}

/// Delayed event scheduler
pub struct DelayedEventScheduler {
    // Priority queue for scheduled events
    scheduled: Arc<parking_lot::Mutex<BinaryHeap<ScheduledEvent>>>,
    
    // Event queue for output
    output_queue: Arc<LockFreeEventQueue>,
    
    // Scheduler state
    running: AtomicBool,
}

#[derive(Clone)]
struct ScheduledEvent {
    event: Event,
    execute_at: Instant,
}

impl PartialEq for ScheduledEvent {
    fn eq(&self, other: &Self) -> bool {
        self.execute_at == other.execute_at
    }
}

impl Eq for ScheduledEvent {}

impl PartialOrd for ScheduledEvent {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        // Reverse order for min-heap behavior
        other.execute_at.partial_cmp(&self.execute_at)
    }
}

impl Ord for ScheduledEvent {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        // Reverse order for min-heap behavior
        other.execute_at.cmp(&self.execute_at)
    }
}

impl DelayedEventScheduler {
    pub fn new(output_capacity: usize) -> Self {
        Self {
            scheduled: Arc::new(parking_lot::Mutex::new(BinaryHeap::new())),
            output_queue: Arc::new(LockFreeEventQueue::new(output_capacity)),
            running: AtomicBool::new(false),
        }
    }
    
    /// Schedule event for future execution
    pub fn schedule(&self, event: Event, delay: Duration) {
        let scheduled = ScheduledEvent {
            event,
            execute_at: Instant::now() + delay,
        };
        
        self.scheduled.lock().push(scheduled);
    }
    
    /// Start scheduler loop
    pub async fn run(&self) {
        self.running.store(true, Ordering::Release);
        
        while self.running.load(Ordering::Acquire) {
            let now = Instant::now();
            let mut ready_events = Vec::new();
            
            // Check for ready events
            {
                let mut heap = self.scheduled.lock();
                while let Some(scheduled) = heap.peek() {
                    if scheduled.execute_at <= now {
                        if let Some(event) = heap.pop() {
                            ready_events.push(event.event);
                        }
                    } else {
                        break;
                    }
                }
            }
            
            // Publish ready events
            for event in ready_events {
                let _ = self.output_queue.push(event);
            }
            
            // Small sleep to prevent busy waiting
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }
    
    /// Stop scheduler
    pub fn stop(&self) {
        self.running.store(false, Ordering::Release);
    }
    
    /// Get output queue
    pub fn output_queue(&self) -> Arc<LockFreeEventQueue> {
        self.output_queue.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_event_queue_priority() {
        let queue = LockFreeEventQueue::new(10);
        
        // Push events with different priorities
        queue.push(Event::new("test".into(), Bytes::from("low"), EventPriority::Low)).unwrap();
        queue.push(Event::new("test".into(), Bytes::from("critical"), EventPriority::Critical)).unwrap();
        queue.push(Event::new("test".into(), Bytes::from("normal"), EventPriority::Normal)).unwrap();
        queue.push(Event::new("test".into(), Bytes::from("high"), EventPriority::High)).unwrap();
        
        // Should pop in priority order
        assert_eq!(queue.pop().unwrap().payload, Bytes::from("critical"));
        assert_eq!(queue.pop().unwrap().payload, Bytes::from("high"));
        assert_eq!(queue.pop().unwrap().payload, Bytes::from("normal"));
        assert_eq!(queue.pop().unwrap().payload, Bytes::from("low"));
    }
    
    #[test]
    fn test_event_router() {
        let router = EventRouter::new(100);
        
        // Subscribe to topics
        let rx1 = router.subscribe("topic1".into());
        let rx2 = router.subscribe("topic2".into());
        
        // Publish events
        router.publish(Event::new("topic1".into(), Bytes::from("msg1"), EventPriority::Normal)).unwrap();
        router.publish(Event::new("topic2".into(), Bytes::from("msg2"), EventPriority::Normal)).unwrap();
        
        // Check received events
        assert_eq!(rx1.try_recv().unwrap().payload, Bytes::from("msg1"));
        assert_eq!(rx2.try_recv().unwrap().payload, Bytes::from("msg2"));
    }
    
    #[tokio::test]
    async fn test_delayed_scheduler() {
        let scheduler = DelayedEventScheduler::new(10);
        
        // Schedule event for 100ms in future
        let event = Event::new("delayed".into(), Bytes::from("test"), EventPriority::Normal);
        scheduler.schedule(event, Duration::from_millis(100));
        
        // Start scheduler
        let scheduler_clone = scheduler.output_queue();
        tokio::spawn(async move {
            scheduler.run().await;
        });
        
        // Wait for event
        tokio::time::sleep(Duration::from_millis(150)).await;
        
        // Check event was executed
        assert!(scheduler_clone.pop().is_some());
    }
}