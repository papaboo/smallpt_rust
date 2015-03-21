//! smallpt in rust. Functionally inspired by smallpt by Kevin Beason.
//!                  http://www.kevinbeason.com/smallpt/

// Warnings go away!
#![feature(env, old_io, old_path, os, std_misc, core)]

use std::num::Float;
use std::ops::{Add, Sub, Mul};
use std::thread;

extern crate rand;

use std::f64::consts::PI;

#[derive(Copy, Clone)]
struct Float3 {
    x: f64,
    y: f64,
    z: f64,
}

impl Float3 {
    fn new(x: f64, y: f64, z: f64) -> Float3 {
        Float3 { x: x, y: y, z: z }
    }

    fn zero() -> Float3 { Float3 { x:0.0, y:0.0, z:0.0 } }

    fn dot(lhs: Float3, rhs: Float3) -> f64 {
        lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z
    }

    fn cross(lhs: Float3, rhs: Float3) -> Float3 {
        Float3 { x: lhs.y * rhs.z - lhs.z * rhs.y,
              y: lhs.z * rhs.x - lhs.x * rhs.z,
              z: lhs.x * rhs.y - lhs.y * rhs.x }
    }

    fn length(self) -> f64 {
        Float3::dot(self, self).sqrt()
    }

    fn normalized(self) -> Float3 {
        let length = self.length();
        Float3 { x: self.x / length, y: self.y / length, z: self.z / length }
    }
}

impl Add for Float3 {
    type Output = Float3;
    fn add(self, rhs: Float3) -> Float3 {
        Float3 { x: self.x + rhs.x,
              y: self.y + rhs.y,
              z: self.z + rhs.z }
    }
}

impl Sub for Float3 {
    type Output = Float3;
    fn sub(self, rhs: Float3) -> Float3 {
        Float3 { x: self.x - rhs.x,
              y: self.y - rhs.y,
              z: self.z - rhs.z }
    }
}

impl Mul for Float3 {
    type Output = Float3;
    fn mul(self, rhs: Float3) -> Float3 {
        Float3 { x: self.x * rhs.x,
              y: self.y * rhs.y,
              z: self.z * rhs.z }
    }
}

impl Mul<f64> for Float3 {
    type Output = Float3;
    fn mul(self, rhs: f64) -> Float3 {
        Float3 { x: self.x * rhs,
              y: self.y * rhs,
              z: self.z * rhs }
    }
}

fn luma(color: Float3) -> f64 {
    0.299 * color.x + 0.587 * color.y + 0.114 * color.z
}

#[derive(Copy, Clone)]
struct Ray {
    origin: Float3,
    direction: Float3,
}

#[derive(Copy, Clone)]
enum BSDF { Diffuse, Mirror, Glass }

#[derive(Copy, Clone)]
struct Sphere {
    radius: f64,
    position: Float3,
    emission: Float3,
    albedo: Float3,
    bsdf: BSDF,
}

impl Sphere {
    fn new(radius: f64, position: Float3, emission: Float3, albedo: Float3, bsdf: BSDF) -> Sphere{
        Sphere { radius: radius, position: position, emission: emission, albedo: albedo, bsdf: bsdf } 
    }
}

// returns distance, infinity if no hit.
fn intersect(ray: Ray, sphere: &Sphere) -> f64{
    // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
    let op: Float3 = sphere.position - ray.origin;
    let b: f64 = Float3::dot(op, ray.direction);
    let det_sqrd: f64 = b * b - Float3::dot(op,op) + sphere.radius * sphere.radius;
    if det_sqrd < 0.0 {
        return Float::infinity();
    }

    let det = det_sqrd.sqrt();
    if b - det > 0.0 {
        b - det
    } else if b + det > 0.0 {
        b + det
    } else {
        Float::infinity()
    }
}

fn clamp01(v: f64) -> f64 { 
    v.max(0.0).min(1.0) 
}

fn tonemap_255(c: f64) -> u8 {
    (clamp01(c).powf(1.0 / 2.2) * 255.0 + 0.5) as u8
}

fn ceil_divide(dividend:usize, divisor:usize) -> usize {
    let division = dividend / divisor;
    if division * divisor == dividend { division } else { division + 1 }
}

fn intersect_scene(ray: Ray, scene: &[Sphere]) -> Option<(&Sphere, f64)> {
    let mut res = (0, Float::infinity());
    for s in 0..scene.len() {
        let t = intersect(ray, &scene[s]);
        let (_, prev_t) = res;
        if t < prev_t && t > 1e-4 {
            res = (s, t);
        }
    }
    
    let (index, t) = res;
    if t == Float::infinity() {
        None
    } else {
        Some((&scene[index], t))
    }
}

fn radiance_estimation(ray: Ray, scene: &[Sphere], depth: i32) -> Float3 {
    let intersection: Option<(&Sphere, f64)> = intersect_scene(ray, scene);
    match intersection {
        None => Float3::zero(),
        Some ((sphere, t)) => {
            let position = ray.origin + ray.direction * t;
            let hard_normal = (position - sphere.position).normalized();
            let forward_normal = if Float3::dot(hard_normal, ray.direction) < 0.0 { hard_normal } else { hard_normal * -1.0 };
            let mut f = sphere.albedo;
            if depth > 3 {
                if rand::random::<f64>() < luma(f) && depth < 100 {
                    f = f * (1.0 / luma(f));
                } else {
                    return sphere.emission;
                }
            }
            
            let irradiance = match sphere.bsdf {
                BSDF::Diffuse => {
                    // Ideal diffuse reflection
                    // Sample cosine distribution and transform into world tangent space.
                    let r1 = 2.0 * PI * rand::random::<f64>();
                    let r2 = rand::random::<f64>();
                    let r2s = r2.sqrt();
                    let wup = if forward_normal.x.abs() > 0.1 { Float3::new(0.0, 1.0, 0.0) } else { Float3::new(1.0, 0.0, 0.0) };
                    let tangent = Float3::cross(forward_normal, wup).normalized();
                    let bitangent = Float3::cross(forward_normal, tangent).normalized(); // Normalize due to precision (although probably not needed when using f64:))
                    let next_direction = tangent * r1.cos() * r2s + bitangent * r1.sin() * r2s + forward_normal * (1.0 - r2).sqrt();
                    radiance_estimation(Ray { origin: position, direction: next_direction.normalized() }, scene, depth+1)
                },

                BSDF::Mirror => {
                    let reflected = ray.direction - forward_normal * 2.0 * Float3::dot(forward_normal, ray.direction);
                    radiance_estimation(Ray { origin: position, direction: reflected }, scene, depth+1)
                },

                BSDF::Glass => {
                    let reflected = ray.direction - forward_normal * 2.0 * Float3::dot(forward_normal, ray.direction);                    
                    let reflection_ray = Ray { origin: position, direction: reflected };
                    // Compute fresnel.
                    let into = Float3::dot(hard_normal, forward_normal) > 0.0;
                    let nc = 1.0;
                    let nt = 1.5;
                    let nnt = if into { nc / nt } else { nt / nc };
                    let ddn = Float3::dot(ray.direction, forward_normal);
                    let cos2t = 1.0 - nnt * nnt * (1.0 - ddn * ddn);
                    if cos2t < 0.0 { // Total internal reflection
                        radiance_estimation(reflection_ray, scene, depth+1)
                    } else {
                        let transmitted_dir = (ray.direction * nnt - hard_normal * (if into {1.0} else {-1.0} * (ddn * nnt + cos2t. sqrt()))).normalized();
                        let transmitted_ray = Ray { origin: position, direction: transmitted_dir };
                        let a = nt-nc;
                        let b = nt + nc;
                        let base_reflectance = a * a / (b * b);
                        let c = 1.0 - if into { -ddn } else { Float3::dot(transmitted_dir, hard_normal) };
                        let reflectance = base_reflectance + (1.0 - base_reflectance) * c * c * c * c * c;
                        let transmittance = 1.0 - reflectance;
                        let rr_propability = 0.25 + 0.5 * reflectance;
                        let reflectance_propability = reflectance / rr_propability;
                        let transmittance_propability = transmittance / (1.0 - rr_propability);
                        if depth > 1 {
                            if rand::random::<f64>() < rr_propability { // Russian roulette between reflectance and transmittance
                                radiance_estimation(reflection_ray, scene, depth+1) * reflectance_propability
                            } else {
                                radiance_estimation(transmitted_ray, scene, depth+1) * transmittance_propability
                            }
                        } else {
                            radiance_estimation(reflection_ray, scene, depth+1) * reflectance + radiance_estimation(transmitted_ray, scene, depth+1) * transmittance
                        }
                    }
                }
            };

            return sphere.emission + f * irradiance;
        }
    }
}

fn main() {
    
    let scene = [Sphere::new(1e5, Float3::new(1e5+1.0,40.8,81.6),      Float3::zero(),              Float3::new(0.75,0.25,0.25),    BSDF::Diffuse),  //Left
                 Sphere::new(1e5, Float3::new(-1e5+99.0,40.8,81.6),    Float3::zero(),              Float3::new(0.25,0.25,0.75),    BSDF::Diffuse),  //Right
                 Sphere::new(1e5, Float3::new(50.0,40.8, 1e5),         Float3::zero(),              Float3::new(0.75,0.75,0.75),    BSDF::Diffuse),  //Back
                 Sphere::new(1e5, Float3::new(50.0,40.8,-1e5+170.0),   Float3::zero(),              Float3::zero(),                 BSDF::Diffuse),  //Front
                 Sphere::new(1e5, Float3::new(50.0, 1e5, 81.6),        Float3::zero(),              Float3::new(0.75,0.75,0.75),    BSDF::Diffuse),  //Bottom
                 Sphere::new(1e5, Float3::new(50.0,-1e5+81.6,81.6),    Float3::zero(),              Float3::new(0.75,0.75,0.75),    BSDF::Diffuse),  //Top
                 Sphere::new(16.5,Float3::new(27.0,16.5,47.0),         Float3::zero(),              Float3::new(0.999,0.999,0.999), BSDF::Mirror),   //Mirror
                 Sphere::new(16.5,Float3::new(73.0,16.5,78.0),         Float3::zero(),              Float3::new(0.999,0.999,0.999), BSDF::Glass),    //Glass
                 Sphere::new(600.0, Float3::new(50.0,681.6-0.27,81.6), Float3::new(12.0,12.0,12.0), Float3::zero(),                 BSDF::Diffuse)]; //Light

    const WIDTH: usize = 512;
    const HEIGHT: usize = 512;
    let samples = match std::env::args().nth(1) {
                      Some(samples_str) => {
                          match samples_str.parse::<i32>() {
                              Ok(s) => s / 4,
                              Err(_) => 1
                          }
                      },
                      None => 1
                  };

    let cam = Ray { origin: Float3::new(50.0, 52.0, 295.6), direction: Float3::new(0.0, -0.042612, -1.0).normalized() };
    let cx = Float3 { x: WIDTH as f64 * 0.5135 / HEIGHT as f64, y: 0.0, z: 0.0 } ;
    let cy = Float3::cross(cx, cam.direction).normalized() * 0.5135;
    
    // Fill backbuffer multithreaded
    let mut backbuffer = std::vec::from_elem(Float3::zero(), WIDTH * HEIGHT);
    {
        let outer_chunks = 100;
        let outer_chunk_size = ceil_divide(WIDTH * HEIGHT, outer_chunks);

        for (outer_chunk_index, outer_chunk) in backbuffer.chunks_mut(outer_chunk_size).enumerate() {
            
            println!("\rRendering ({} spp) {}%", samples*4, outer_chunk_index);

            let inner_chunks = std::os::num_cpus() * std::os::num_cpus();
            let inner_chunk_size = ceil_divide(outer_chunk_size, inner_chunks);
            
            // Create and launch threads. Automatically joins when going out of scope.
            let mut threadscope = std::vec::Vec::with_capacity(inner_chunks);
            for (inner_chunk_index, inner_chunk) in outer_chunk.chunks_mut(inner_chunk_size).enumerate() {
                threadscope.push(thread::scoped(move || {
                    for i in 0..inner_chunk.len() {
                        let pixel_index = i + inner_chunk_index * inner_chunk_size + outer_chunk_index * outer_chunk_size;
                        let x = pixel_index % WIDTH;
                        let y = HEIGHT - pixel_index / WIDTH - 1;
                        
                        let mut radiance = Float3::zero();
                        // Sample 2x2 subpixels.
                        for sy in 0..2 {
                            for sx in 0..2 {
                                // Samples per subpixel.
                                for _ in 0..samples {
                                    let r1:f64 = 2.0 * rand::random::<f64>();
                                    let dx = if r1 < 1.0 { r1.sqrt() - 1.0 } else { 1.0 - (2.0 - r1).sqrt() };
                                    let r2:f64 = 2.0 * rand::random::<f64>();
                                    let dy = if r2 < 1.0 { r2.sqrt() - 1.0 } else { 1.0 - (2.0 - r2).sqrt() };
                                    let view_dir = cam.direction + cx * (((sx as f64 + 0.5 + dx) / 2.0 + x as f64) / WIDTH as f64 - 0.5) +
                                        cy * (((sy as f64 + 0.5 + dy) / 2.0 + y as f64) / HEIGHT as f64 - 0.5);
                                    let ray = Ray { origin: cam.origin + view_dir * 130.0, direction: view_dir.normalized() };
                                    radiance = radiance + radiance_estimation(ray, &scene, 0);
                                }
                            }
                        }
                        
                        inner_chunk[i] = radiance * (0.25 / samples as f64);
                    }
                }))
            }
        }
    }

    // Create PPM file content.
    let mut ppm: String = format!("P3\n{} {}\n{}\n", WIDTH, HEIGHT, 255);
    for p in 0..WIDTH*HEIGHT {
        let pixel = backbuffer[p];
        let rgb_string = format!("{} {} {} ", tonemap_255(pixel.x), tonemap_255(pixel.y), tonemap_255(pixel.z));
        ppm = ppm + &rgb_string;
    }

    // Write content to PPM file.
    let image_path = Path::new("image.ppm");
    let mut file = match std::old_io::File::create(&image_path) {
        Err(why) => panic!("couldn't create image.ppm: {}", why.desc),
        Ok(file) => file,
    };
    
    match file.write_str(&ppm) {
        Err(why) => panic!("couldn't write to image.ppm: {}", why.desc),
        Ok(_) => {},
    };
}
