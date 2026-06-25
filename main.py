import yaml
from recovery_agent.agent import RecoveryAgent

def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    agent = RecoveryAgent(config)
    
    test_pdb = "broken_test.pdb" 
    
    agent.run(test_pdb)

if __name__ == "__main__":
    main()
